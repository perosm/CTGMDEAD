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
        self.detection_head = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels_detection
            * number_of_object_proposals_per_anchor,
            kernel_size=1,
        )
        self.regression_head = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels_regression
            * number_of_object_proposals_per_anchor,
            kernel_size=1,
        )
        # self._init_weights()

    # def _init_weights(self):
    #     """
    #     In the original paper they initialize the weights of the RPN module by drawing
    #     weights from a zero-mean Gaussian distribution with standard deviation 0.01.
    #     """
    #     for module in self.modules():
    #         if isinstance(module, nn.Conv2d):
    #             nn.init.normal_(module.weight, mean=0, std=0.01)
    #             if module.bias is not None:
    #                 nn.init.constant_(module.bias, 0)

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
            N,
            self.out_channels_detection,
            self.number_of_object_proposals_per_anchor,
            H_fmap,
            W_fmap,
        )
        objectness_score = objectness_score.permute(0, 2, 3, 4, 1)
        objectness_score = objectness_score.view(N, -1)
        bbox_regression_deltas = bbox_regression_deltas.view(
            N,
            self.out_channels_regression,
            self.number_of_object_proposals_per_anchor,
            H_fmap,
            W_fmap,
        )
        bbox_regression_deltas = bbox_regression_deltas.permute(0, 2, 3, 4, 1)
        bbox_regression_deltas = bbox_regression_deltas.view(
            N, -1, self.out_channels_regression
        )

        return objectness_score, bbox_regression_deltas

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        objectness_logits = self.detection_head(x)
        bbox_regression_deltas = self.regression_head(x)

        objectness_logits, bbox_regression_deltas = (
            self._format_objectness_and_bbox_regression_deltas(
                objectness_score=objectness_logits,
                bbox_regression_deltas=bbox_regression_deltas,
            )
        )
        return objectness_logits, bbox_regression_deltas


class RegionProposalNetwork(nn.Module):

    def __init__(self, configs: list[dict], feature_map_names: list[str]) -> None:
        """
        Region Proposal Network (RPN) is used to predict whether an object exists
        """
        super().__init__()
        self.feature_map_names = feature_map_names
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
        self.rpn_heads = self._configure_rpn_heads(rpn_head_config=configs["rpn_head"])

    def _configure_shared_conv_layer(
        self, num_channels: int, num_fpn_outputs: int
    ) -> nn.Module:
        return nn.ModuleDict(
            {
                f"fpn{i}": nn.Sequential(
                    nn.Conv2d(
                        in_channels=num_channels,
                        out_channels=num_channels,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.BatchNorm2d(num_features=num_channels),
                    nn.ReLU(),
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

    def _configure_rpn_heads(self, rpn_head_config: list[dict]) -> RPNHead:
        rpn_head_config = list_of_dict_to_dict(
            list_of_dicts=rpn_head_config, new_dict={}, depth_cnt=1
        )
        number_of_object_proposals_per_anchor = rpn_head_config[
            "number_of_object_proposals_per_anchor"
        ]
        return nn.ModuleDict(
            {
                name: RPNHead(
                    number_of_object_proposals_per_anchor=number_of_object_proposals_per_anchor
                )
                for name in self.feature_map_names
            }
        )

    def _filter_proposals(
        self,
        proposals: torch.Tensor,  # (num_anchors, 4)
        objectness_logits: torch.Tensor,  # (num_anchors, 2)
    ):
        """
        Works only for batchsize=1.

        Proposals per feature map are being filtered in the following order:
            - 1) Clip proposals to fit into image
            - 2) Proposals with objectness_score < objectness_threshold
            - 3) Filtering before applying Non-Max Supression (NMS)
            - 4) Non-Max Supression (NMS) between proposals for the same ground truth
            - 5) Pick top K proposals

        Args:
        """
        # 1) clip proposals
        objectness_logits = objectness_logits.detach()
        proposals = clip_boxes_to_image(boxes=proposals, size=self.image_size)

        # 2) filter proposals with objectness_score < objectness_threshold
        objectness_score = torch.sigmoid(objectness_logits)
        keep = objectness_score > self.objectness_threshold
        proposals = proposals[keep]
        objectness_logits = objectness_logits[keep]

        # 3) Filter before applying NMS
        _, indices = torch.topk(
            objectness_logits,
            k=min(self.pre_nms_filtering, objectness_logits.numel()),
            largest=True,
        )
        proposals = proposals[indices]
        objectness_logits = objectness_logits[indices]

        # 4) filter using NMS
        keep = nms(
            boxes=proposals,
            scores=objectness_logits,
            iou_threshold=self.iou_threshold,
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
        objectness_logits = objectness_logits[keep]

        return objectness_logits, proposals

    def forward(
        self, fpn_feature_map_outputs: dict[str, torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        (
            all_anchors,
            all_objectness_scores,
            all_bbox_regression_deltas,
            filtered_objectness_logits,
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
            objectness_logits, bbox_regression_deltas = self.rpn_heads[
                fpn_feature_map_name
            ](intermediary)
            objectness_logits = objectness_logits.squeeze(0)
            bbox_regression_deltas = bbox_regression_deltas.squeeze(0)
            anchors = clip_boxes_to_image(anchors, self.image_size)
            proposals = apply_deltas_to_boxes(
                boxes=anchors, deltas=bbox_regression_deltas.detach()
            )
            objectness_logits_filtered, proposals_filtered = self._filter_proposals(
                proposals=proposals,
                objectness_logits=objectness_logits,
            )
            all_anchors.append(anchors)
            all_objectness_scores.append(objectness_logits)
            all_bbox_regression_deltas.append(bbox_regression_deltas)
            filtered_objectness_logits.append(objectness_logits_filtered)
            filtered_proposals.append(proposals_filtered)

        return (
            torch.cat(all_anchors, dim=0),
            torch.cat(all_objectness_scores, dim=0),
            torch.cat(all_bbox_regression_deltas, dim=0),
            torch.cat(filtered_objectness_logits, dim=0),
            torch.cat(filtered_proposals, dim=0),
        )
