import torch
from torch import nn
from torchvision.ops import clip_boxes_to_image, nms
from model.object_detection.anchor_generator import AnchorGenerator

from utils.shared.dict_utils import list_of_dict_to_dict
from utils.object_detection.utils import apply_deltas_to_boxes


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
        self._init_weights()

    def _init_weights(self):
        """
        In the original paper they initialize the weights of the RPN module by drawing
        weights from a zero-mean Gaussian distribution with standard deviation 0.01.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0, std=0.01)
                if module.bias is not None:
                    nn.init.normal_(module.bias, mean=0, std=0.01)

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

    def __init__(
        self,
        configs: list[dict],
    ) -> None:
        """
        Region Proposal Network (RPN) is used to predict whether an object exists
        """
        super().__init__()
        self.training = configs["training"]
        self.image_size = configs["image_size"]
        self.objectness_threshold = configs["objectness_threshold"]
        self.iou_threshold = configs["iou_threshold"]
        self.top_k_proposals_training = configs["top_k_proposals_training"]
        self.top_k_proposals_testing = configs["top_k_proposals_testing"]
        self.pre_nms_filtering = configs["pre_nms_filtering"]

        self.anchor_generator = self._configure_anchor_generator(
            anchor_generator_config=configs["anchor_generator"]
        )
        self.conv = self._configure_shared_conv_layer(
            num_channels=configs["num_channels"],
            num_fpn_outputs=configs["num_fpn_outputs"],
        )
        self.rpn_head = self._configure_rpn_head(rpn_head_config=configs["rpn_head"])

    def _configure_shared_conv_layer(
        self, num_channels: int, num_fpn_outputs: int
    ) -> nn.Module:
        return nn.ModuleDict(
            {
                f"fpn{i}": nn.Conv2d(
                    in_channels=num_channels,
                    out_channels=num_channels,
                    kernel_size=3,
                    padding=1,
                )
                for i in range(num_fpn_outputs)
            }
        )

    def _configure_anchor_generator(
        self, anchor_generator_config: list[dict]
    ) -> AnchorGenerator:
        anchor_generator_config = list_of_dict_to_dict(
            list_of_dicts=anchor_generator_config, new_dict={}, depth_cnt=1
        )
        sizes = anchor_generator_config["sizes"]
        aspect_ratios = anchor_generator_config["aspect_ratios"]

        return AnchorGenerator(aspect_ratios=aspect_ratios, sizes=sizes)

    def _configure_rpn_head(self, rpn_head_config: list[dict]) -> RPNHead:
        rpn_head_config = list_of_dict_to_dict(
            list_of_dicts=rpn_head_config, new_dict={}, depth_cnt=1
        )
        number_of_object_proposals_per_anchor = rpn_head_config[
            "number_of_object_proposals_per_anchor"
        ]
        return RPNHead(
            number_of_object_proposals_per_anchor=number_of_object_proposals_per_anchor
        )

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
        x1, y1, x2, y2 = anchors.unbind(dim=-1)  # left, top, right, bottom
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
            - 2) Proposals with objectness_score < objectness_threshold
            - 3) Filtering before applying Non-Max Supression (NMS)
            - 4) Non-Max Supression (NMS) between proposals for the same ground truth
            - 5) Pick top K proposals

        Args:
        """
        # TODO: Make it work for N > 1
        # 1) clip proposals
        proposals = clip_boxes_to_image(boxes=proposals, size=self.image_size)

        # 2) filter proposals with objectness_score < objectness_threshold
        keep = objectness_score > self.objectness_threshold
        proposals = proposals[keep]
        objectness_score = objectness_score[keep]

        # 3) Filter before applying NMS
        _, indices = torch.topk(
            objectness_score,
            k=min(self.pre_nms_filtering, objectness_score.numel()),
            largest=True,
        )
        proposals = proposals[indices]
        objectness_score = objectness_score[indices]

        # 4) filter using NMS
        keep = nms(
            boxes=proposals, scores=objectness_score, iou_threshold=self.iou_threshold
        )

        # 5) pick top K proposals
        top_k = min(
            (
                self.top_k_proposals_training
                if self.training
                else self.top_k_proposals_testing
            ),
            keep.numel(),
        )
        keep = keep[:top_k]

        proposals = proposals[keep]
        objectness_score = objectness_score[keep]

        return objectness_score, proposals

    def forward(
        self, fpn_feature_map_outputs: dict[str, torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        (
            all_anchors,
            all_objectness_scores,
            all_bbox_regression_deltas,
            filtered_objectness_scores,
            filtered_proposals,
        ) = ([], [], [], [], [])
        anchors_generated, num_anchors_per_feature_map = self.anchor_generator(
            self.image_size, list(fpn_feature_map_outputs.values())
        )
        # for each feature map H_fm x W_fm anchors are created
        anchors_per_feature_map = torch.split(
            anchors_generated, num_anchors_per_feature_map
        )
        for fpn_feature_map_name, anchors in zip(
            fpn_feature_map_outputs, anchors_per_feature_map
        ):
            intermediary = self.conv[fpn_feature_map_name](
                fpn_feature_map_outputs[fpn_feature_map_name]
            )
            objectness_score, bbox_regression_deltas = self.rpn_head(
                intermediary
            )  # TODO: create separate RPN heads for each fpn?
            # we don't want to keep track of computational graph when applying transformation
            # to anchors because we don't want to backprop loss from RPN module ?
            anchors = clip_boxes_to_image(anchors, self.image_size)
            proposals = apply_deltas_to_boxes(
                boxes=anchors, deltas=bbox_regression_deltas.detach()
            )
            filtered_objectness_score, proposals = self._filter_proposals(
                proposals=proposals,
                objectness_score=objectness_score,
            )
            all_anchors.append(anchors)
            all_objectness_scores.append(objectness_score)
            filtered_objectness_scores.append(filtered_objectness_score)
            filtered_proposals.append(proposals)
            all_bbox_regression_deltas.append(bbox_regression_deltas)

        return (
            torch.cat(all_anchors, dim=0),
            torch.cat(all_objectness_scores, dim=1),
            torch.cat(filtered_objectness_scores),
            torch.cat(filtered_proposals),
            torch.cat(all_bbox_regression_deltas, dim=1),
        )
