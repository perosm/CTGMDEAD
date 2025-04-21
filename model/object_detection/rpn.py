import torch
from torch import nn
from torchvision.ops import clip_boxes_to_image, nms


class RPNHead(nn.Module):
    in_channels = 256
    out_channels_detection = 1
    out_channels_regression = 4

    def __init__(self, number_of_object_proposals_per_anchor: int = 3):
        """
        Args:
            - num_object_proposals_per_anchor: Number of object proposals per anchor. By default
                                               value of 3 is used (1 bounding box size * 3 aspect ratios)
        """
        super().__init__()
        self.number_of_object_proposals_per_anchor = (
            number_of_object_proposals_per_anchor
        )
        self.detection_head = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels_detection
                * number_of_object_proposals_per_anchor,
                kernel_size=1,
            ),
            nn.Sigmoid(),
        )
        self.regression_head = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels_regression
            * number_of_object_proposals_per_anchor,
            kernel_size=1,
        )

    def _format_objectness_and_bbox_regression_deltas(
        self, objectness_score: torch.Tensor, bbox_regression_deltas: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Formats both objectness_score and bbox_regression_deltas accordingly.

        Args:
            - objectness_score: Objectness score tensor of shape (N, num_object_proposals_per_anchor, H_fmap, W_fmap).
            - bbox_regression_deltas: Bounding box regression output of shape (N, 4 * num_object_proposals_per_anchor, H_fmap, W_fmap).

        Returns:
            - objectness_score: Objectness score tensor of shape (N, num_object_proposals_per_anchor * H_fmap * W_fmap).
            - bbox_regression_deltas: Bounding box regression output of shape (N, 4 * num_object_proposals_per_anchor, H_fmap, W_fmap).
        """
        N, _, H_fmap, W_fmap = objectness_score.shape

        objectness_score = objectness_score.view(
            N, self.number_of_object_proposals_per_anchor * H_fmap * W_fmap
        )
        bbox_regression_deltas = bbox_regression_deltas.view(
            N, -1, self.number_of_object_proposals_per_anchor, H_fmap, W_fmap
        )
        bbox_regression_deltas = bbox_regression_deltas.permute(0, 2, 3, 4, 1)
        bbox_regression_deltas = bbox_regression_deltas.view(N, -1, 4)

        return objectness_score, bbox_regression_deltas

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        N = x.shape[0]
        objectness_score = self.detection_head(x)
        bbox_regression_deltas = self.regression_head(x)

        objectness_score, bbox_regression_deltas = (
            self._format_objectness_and_bbox_regression_deltas(
                objectness_score=objectness_score,
                bbox_regression_deltas=bbox_regression_deltas,
            )
        )
        return objectness_score, bbox_regression_deltas


class RegionProposalNetwork(nn.Module):
    out_channels = 256

    def __init__(
        self,
        num_channels_per_feature_map: list[int],
        anchor_generator: AnchorGenerator,
        training: bool,
        image_size: tuple[int, int],
        objectness_threshold: float = 0.3,
        iou_threshold: float = 0.7,
        top_k_proposals_training: int = 2000,
        top_k_proposals_testing: int = 300,
    ) -> None:
        """
        Region Proposal Network (RPN) is used to predict whether an object exists
        """
        super().__init__()
        self.conv = nn.ModuleDict(
            {
                f"fpn_{i+1}": nn.Conv2d(
                    in_channels=num_channels,
                    out_channels=self.out_channels,
                    kernel_size=3,
                    padding=1,
                )
                for i, num_channels in enumerate(num_channels_per_feature_map)
            }
        )
        self.rpn_head = RPNHead()
        self.anchor_generator = anchor_generator
        self.training = training
        self.image_size = image_size
        self.objectness_threshold = objectness_threshold
        self.iou_threshold = iou_threshold
        self.top_k_proposals_training = top_k_proposals_training
        self.top_k_proposals_testing = top_k_proposals_testing

    def _fetch_proposals(
        self, anchors: torch.Tensor, bbox_regression_deltas: torch.Tensor
    ) -> torch.Tensor:
        """
        Fetches proposals given anchors and bbox_regression_deltas for each of the anchors.

        Args:
            - anchors:
            - bbox_regression_deltas:

        Returns:
            - proposals:
        """
        # corners of the anchors
        x1, y1, x2, y2 = anchors.unbind(dim=-1)
        anchor_width = x2 - x1
        anchor_height = y2 - y1
        anchor_center_x = y1 + anchor_width / 2
        anchor_center_y = x1 + anchor_height / 2

        deltas_x, deltas_y, deltas_w, deltas_h = bbox_regression_deltas.unbind(dim=-1)

        # center coordinates and height, width of predicted bounding box
        pred_x = anchor_center_x + deltas_x * anchor_width
        pred_y = anchor_center_y + deltas_y * anchor_height
        pred_width = anchor_width * torch.exp(deltas_w)
        pred_height = anchor_height * torch.exp(deltas_h)

        # switching back to corners of predicted bounding box
        pred_x1 = pred_x - pred_width / 2
        pred_y1 = pred_y - pred_height / 2
        pred_x2 = pred_x + pred_width / 2
        pred_y2 = pred_y + pred_height / 2

        proposals = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=-1)

        return proposals

    def _filter_proposals(
        self,
        proposals: torch.Tensor,
        objectness_score: torch.Tensor,
    ):
        """
        Proposals per feature map are being filtered in the following order:
            - 1) Clip proposals to fit into image
            - 2) Filter degenerate proposals TODO: this should not be neccessary?
            - 3) Proposals with objectness_score < objectness_threshold
            - 4) Non-Max Supression (NMS) between proposals for the same ground truth
            - 5) Pick top K proposals

        Args:
        """
        # TODO: Make it work for N > 1
        N = objectness_score.shape[0]
        # 1) clip proposals
        proposals = clip_boxes_to_image(boxes=proposals, size=self.image_size)
        proposals = proposals.repeat(N, 1, 1)

        # 2) filter degenerate proposals after clipping
        proposal_heights = proposals[..., 2] - proposals[..., 0]
        proposal_widths = proposals[..., 3] - proposals[..., 1]
        keep = (proposal_heights > 0) & (proposal_widths > 0)
        proposals = proposals[keep]
        objectness_score = objectness_score[keep]

        # 3) filter proposals with objectness_score < objectness_threshold
        keep = objectness_score > self.objectness_threshold
        proposals = proposals[keep]
        objectness_score = objectness_score[keep]

        # 4) filter using NMS
        keep = nms(
            boxes=proposals, scores=objectness_score, iou_threshold=self.iou_threshold
        )
        proposals = proposals[keep]
        objectness_score = objectness_score[keep]

        # 5) pick top K proposals
        top_k = (
            self.top_k_proposals_training
            if self.training
            else self.top_k_proposals_testing
        )
        _, indices = torch.topk(
            objectness_score, k=min(top_k, objectness_score.numel())
        )

        proposals = proposals[indices]
        objectness_score = objectness_score[indices]

        return objectness_score, proposals

    def forward(
        self,
        fpn_feature_map_outputs: dict[str, torch.Tensor],
        y_true: torch.Tensor | None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        objectness_score_per_feature_map, proposals_per_feature_map = {}, {}
        all_anchors, num_anchors_per_feature_map = self.anchor_generator(
            self.image_size, list(fpn_feature_map_outputs.values())
        )
        # for each feature map H_fm x W_fm anchors are created
        anchors_per_feature_map = torch.split(all_anchors, num_anchors_per_feature_map)
        for fpn_feature_map_name, anchors in zip(
            fpn_feature_map_outputs, anchors_per_feature_map
        ):
            intermediary = self.conv[fpn_feature_map_name](
                fpn_feature_map_outputs[fpn_feature_map_name]
            )
            objectness_score, bbox_regression_deltas = self.rpn_head(
                intermediary
            )  # TODO: create separate RPN heads for each fpn?
            proposals = self._fetch_proposals(
                anchors=anchors, bbox_regression_deltas=bbox_regression_deltas
            )
            objectness_score, proposals = self._filter_proposals(
                proposals=proposals,
                objectness_score=objectness_score,
            )
            objectness_score_per_feature_map[fpn_feature_map_name] = objectness_score
            proposals_per_feature_map[fpn_feature_map_name] = proposals

        return objectness_score_per_feature_map, proposals_per_feature_map


if __name__ == "__main__":
    DEVICE = "cuda"
    num_channels_per_feature_map = [64, 128, 256, 512]
    anchor_generator = AnchorGenerator(
        sizes=(  # TODO: which anchor sizes to choose?
            # (64),  # fpn_1
            (128),  # fpn_2
            (256),  # fpn_3
            (512),  # fpn_4
        ),
        aspect_ratios=(
            # (0.5, 1.0, 2.0),
            (0.5, 1.0, 2.0),
            (0.5, 1.0, 2.0),
            (0.5, 1.0, 2.0),
        ),
    ).to(DEVICE)
    input_image = torch.zeros((1, 3, 256, 1184)).to(DEVICE)
    feature_maps = {
        # "fpn_1": torch.zeros((1, 64, 128, 592)).to(DEVICE),
        "fpn_2": torch.zeros((1, 128, 64, 296)).to(DEVICE),
        "fpn_3": torch.zeros((1, 256, 32, 148)).to(DEVICE),
        "fpn_4": torch.zeros((1, 512, 16, 74)).to(DEVICE),
    }
    rpn = RegionProposalNetwork(
        num_channels_per_feature_map=num_channels_per_feature_map,
        anchor_generator=anchor_generator,
        image_size=input_image.shape[-2:],
        training=True,
    ).to(DEVICE)
    y_true = torch.arange(15).to(DEVICE)
    rpn(feature_maps, y_true)
