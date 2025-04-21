import torch
from torch import nn
from torchvision.ops import clip_boxes_to_image, remove_small_boxes, nms
from torchvision.models.detection.rpn import concat_box_prediction_layers


class AnchorGenerator(nn.Module):
    """
    Module that generates anchors for a set of feature maps and
    image sizes.

    The module support computing anchors at multiple sizes and aspect ratios
    per feature map. This module assumes aspect ratio = height / width for
    each anchor.

    sizes and aspect_ratios should have the same number of elements, and it should
    correspond to the number of feature maps.

    sizes[i] and aspect_ratios[i] can have an arbitrary number of elements,
    and AnchorGenerator will output a set of sizes[i] * aspect_ratios[i] anchors
    per spatial location for feature map i.

    Args:
        sizes (Tuple[Tuple[int]]):
        aspect_ratios (Tuple[Tuple[float]]):
    """

    __annotations__ = {
        "cell_anchors": list[torch.Tensor],
    }

    def __init__(
        self,
        sizes=((128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),),
    ):
        super().__init__()

        if not isinstance(sizes[0], (list, tuple)):
            # TODO change this
            sizes = tuple((s,) for s in sizes)
        if not isinstance(aspect_ratios[0], (list, tuple)):
            aspect_ratios = (aspect_ratios,) * len(sizes)

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = [
            self.generate_anchors(size, aspect_ratio)
            for size, aspect_ratio in zip(sizes, aspect_ratios)
        ]

    # TODO: https://github.com/pytorch/pytorch/issues/26792
    # For every (aspect_ratios, scales) combination, output a zero-centered anchor with those values.
    # (scales, aspect_ratios) are usually an element of zip(self.scales, self.aspect_ratios)
    # This method assumes aspect ratio = height / width for an anchor.
    def generate_anchors(
        self,
        scales: list[int],
        aspect_ratios: list[float],
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        scales = torch.as_tensor(scales, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios

        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)

        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        return base_anchors.round()

    def set_cell_anchors(self, dtype: torch.dtype, device: torch.device):
        self.cell_anchors = [
            cell_anchor.to(dtype=dtype, device=device)
            for cell_anchor in self.cell_anchors
        ]

    def num_anchors_per_location(self) -> list[int]:
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]

    # For every combination of (a, (g, s), i) in (self.cell_anchors, zip(grid_sizes, strides), 0:2),
    # output g[i] anchors that are s[i] distance apart in direction i, with the same dimensions as a.
    def grid_anchors(
        self, grid_sizes: list[list[int]], strides: list[list[torch.Tensor]]
    ) -> list[torch.Tensor]:
        anchors = []
        cell_anchors = self.cell_anchors
        torch._assert(cell_anchors is not None, "cell_anchors should not be None")
        torch._assert(
            len(grid_sizes) == len(strides) == len(cell_anchors),
            "Anchors should be Tuple[Tuple[int]] because each feature "
            "map could potentially have different sizes and aspect ratios. "
            "There needs to be a match between the number of "
            "feature maps passed and the number of sizes / aspect ratios specified.",
        )

        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
            grid_height, grid_width = size
            stride_height, stride_width = stride
            device = base_anchors.device

            # For output anchor, compute [x_center, y_center, x_center, y_center]
            shifts_x = (
                torch.arange(0, grid_width, dtype=torch.int32, device=device)
                * stride_width
            )
            shifts_y = (
                torch.arange(0, grid_height, dtype=torch.int32, device=device)
                * stride_height
            )
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            # For every (base anchor, output anchor) pair,
            # offset each zero-centered base anchor by the center of the output anchor.
            anchors.append(
                (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
            )

        return anchors

    def forward(
        self, image_size: tuple[int, int], feature_maps: list[torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        strides = [
            [
                torch.empty((), dtype=torch.int64, device=device).fill_(
                    image_size[0] // g[0]
                ),
                torch.empty((), dtype=torch.int64, device=device).fill_(
                    image_size[1] // g[1]
                ),
            ]
            for g in grid_sizes
        ]
        self.set_cell_anchors(dtype, device)
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes, strides)
        num_anchors_per_feature_map = [
            anchors_per_feature_map.shape[0]
            for anchors_per_feature_map in anchors_over_all_feature_maps
        ]

        anchors = torch.cat(anchors_over_all_feature_maps)
        return anchors, num_anchors_per_feature_map


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
