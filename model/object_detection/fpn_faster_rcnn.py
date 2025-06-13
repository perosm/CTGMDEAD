import torch
from torch import nn
from torchvision.ops import remove_small_boxes, clip_boxes_to_image, batched_nms
from model.object_detection.rpn import RegionProposalNetwork
from model.object_detection.roi import ROINetwork
from model.object_detection.output_heads import OutputHeads
from model.object_detection_3d.distance_head import DistanceHead
from model.object_detection_3d.attribute_head import AttributeHead
from utils.shared.dict_utils import list_of_dict_to_dict
from utils.object_detection.utils import apply_deltas_to_boxes
from utils.shared.enums import TaskEnum


class FPNFasterRCNNLinkerBlock(nn.Module):
    def __init__(
        self, num_channels_per_feature_map: list[int], out_channels: int = 256
    ) -> None:
        """
        When using Faster R-CNN with Feature Pyramid Network (FPN) a problem occurs if we want to reuse the MultiScaleRoIAlign
        https://pytorch.org/vision/main/generated/torchvision.ops.MultiScaleRoIAlign.html.
        In order to fix that, a so called "Linker block" is added to "link" FPN with Faster R-CNN part of the network.

        Args:
            - num_channels_per_feature_map: List of ints representing number of channels per FPN output.
            - out_channels: Integer representing number of output channels after Linker block. Default value is 256.
        """
        super().__init__()
        self.num_channels_per_feature_map = num_channels_per_feature_map
        self.out_channels = out_channels
        self.linker_blocks = nn.ModuleDict(
            {
                f"fpn{i}": nn.Sequential(
                    nn.Conv2d(
                        in_channels=num_channels,
                        out_channels=self.out_channels,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.BatchNorm2d(num_features=self.out_channels),
                    nn.ReLU(),
                )
                for i, num_channels in enumerate(self.num_channels_per_feature_map)
            }
        )

    def forward(self, fpn_outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        linked_fpn_outputs = {}
        fpn_names = fpn_outputs.keys()
        for fpn_name in fpn_names:
            linked_fpn_outputs[fpn_name] = self.linker_blocks[fpn_name](
                fpn_outputs[fpn_name]
            )

        return linked_fpn_outputs


class FPNFasterRCNN(nn.Module):
    def __init__(self, configs: dict):
        super().__init__()
        self.image_size = configs["image_size"]
        self.pool_output_size = configs["pool_output_size"]
        self.feature_map_names = configs["feature_map_names"]
        self.num_channels_per_feature_map = configs["num_channels_per_feature_map"]
        self.out_channels = configs["out_channels"]
        self.probability_threshold = configs["probability_threshold"]
        self.iou_threshold = configs["iou_threshold"]
        self.num_classes = configs["num_classes"]
        self.linker_layer = self._configure_linker_layer(
            num_channels_per_feature_map=self.num_channels_per_feature_map,
            out_channels=self.out_channels,
        )
        # Shared
        self.rpn = self._configure_region_proposal_network(rpn_config=configs["rpn"])
        self.roi = self._configure_region_of_interest_network(roi_config=configs["roi"])

        # Object detection 2D
        self.output_heads = self._configure_output_heads(
            output_heads_config=configs["output_heads"]
        )

        # Object detection 3D
        self.distance_head = self._configure_distance_head(
            distance_head_config=configs["distance_head"]
        )
        self.attribute_head = self._configure_attribute_head(
            attribute_head_config=configs["attribute_head"]
        )

    def _configure_linker_layer(
        self, num_channels_per_feature_map: list[int], out_channels: int
    ) -> FPNFasterRCNNLinkerBlock:
        return FPNFasterRCNNLinkerBlock(
            num_channels_per_feature_map=num_channels_per_feature_map,
            out_channels=out_channels,
        )

    def _configure_region_proposal_network(
        self, rpn_config: list[dict]
    ) -> RegionProposalNetwork:
        rpn_config.append({"image_size": self.image_size})
        rpn_config.append({"num_fpn_outputs": len(self.num_channels_per_feature_map)})
        rpn_config.append({"num_channels": self.out_channels})
        rpn_config = list_of_dict_to_dict(
            list_of_dicts=rpn_config, new_dict={}, depth_cnt=1
        )
        return RegionProposalNetwork(
            configs=rpn_config, feature_map_names=self.feature_map_names
        )

    def _configure_region_of_interest_network(
        self, roi_config: list[dict]
    ) -> ROINetwork:
        roi_config = list_of_dict_to_dict(roi_config, new_dict={}, depth_cnt=1)
        return ROINetwork(
            image_size=self.image_size,
            pool_output_size=self.pool_output_size,
            feature_map_names=self.feature_map_names,
            sampling_ratio=roi_config["sampling_ratio"],
        )

    def _configure_output_heads(self, output_heads_config: list[dict]) -> OutputHeads:
        output_heads_config = list_of_dict_to_dict(
            list_of_dicts=output_heads_config,
        )
        return OutputHeads(
            image_size=self.image_size,
            pool_size=self.pool_output_size,
            num_classes=self.num_classes,
        )

    def _configure_distance_head(
        self, distance_head_config: list[dict]
    ) -> DistanceHead:
        distance_head_config = list_of_dict_to_dict(
            list_of_dicts=distance_head_config, new_dict={}, depth_cnt=1
        )
        return DistanceHead(
            num_conv_layers=distance_head_config["num_conv_layers"],
            num_channels=distance_head_config["num_channels"],
            num_fc_layers=distance_head_config["num_fc_layers"],
            rpn_output_channels=self.out_channels,
            pool_output_size=self.pool_output_size,
            fc_features=distance_head_config["fc_features"],
        )

    def _configure_attribute_head(
        self, attribute_head_config: list[dict]
    ) -> AttributeHead:
        attribute_head_config = list_of_dict_to_dict(
            list_of_dicts=attribute_head_config, new_dict={}, depth_cnt=1
        )
        return AttributeHead(
            num_conv_layers=attribute_head_config["num_conv_layers"],
            num_channels=attribute_head_config["num_channels"],
            num_fc_layers=attribute_head_config["num_fc_layers"],
            fc_features=attribute_head_config["fc_features"],
            rpn_output_channels=self.out_channels,
            pool_output_size=self.pool_output_size,
            num_classes=self.num_classes,
        )

    def _fetch_probabilities_boxes_and_labels(
        self,
        proposals: torch.Tensor,
        bbox_regression_deltas: torch.Tensor,
        class_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Selects the best bounding box deltas per proposal based on predicted class labels,
        then applies deltas to proposals to generate final bounding boxes.

        Args:
            proposals: Filtered proposals in a single image of shape (num_proposals, 4).
            bbox_regression_deltas: Deltas for all classes, shape (num_proposals, num_classes * 4).
            class_logits: Class logits of shape (num_proposals, num_classes).

        Returns:
            Class probabilities of shape (num_proposals).
            Bounding boxes of shape (num_proposals, 4).
            Labels of for each of the bounding boxes (num_proposals).
        """

        class_probits = torch.softmax(class_logits, dim=-1)
        labels = torch.argmax(class_probits, dim=-1)
        class_probits = class_probits[torch.arange(class_probits.shape[0]), labels]
        bbox_regression_deltas = bbox_regression_deltas.view(
            -1, class_logits.shape[-1], 4
        )
        bbox_regression_deltas = torch.gather(
            bbox_regression_deltas,
            1,
            labels.view(-1, 1, 1).expand(-1, 1, 4),
        ).squeeze(1)
        bounding_boxes = apply_deltas_to_boxes(
            boxes=proposals, deltas=bbox_regression_deltas
        )

        return class_probits, bounding_boxes, labels

    def _filter_boxes(
        self,
        bounding_boxes: torch.Tensor,
        class_probits: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            bounding_boxes: Predicted bounding boxes (num_boxes, 4).
            class_probits: Predicted class logits (num_boxes, num_classes)
            labels: Predicted class labels (num_boxes).

        Final predicted boxes are filtered in the next few steps:
            - 1) Clip boxes that go outside of frame.
            - 2) Remove small boxes.
            - 3) Remove boxes with background labels.
            - 4) Remove boxes with small probability.
            - 5) Remove overlapping objects of same class using Non-max suppresion (NMS).

        Returns:
            bounding_boxes: Filtered bounding boxes tensor (filtered_num_boxes, 4).
            class_probits: Filtered class probability for each of the bounding boxes (filtered_num_boxes).
            class_labels: Filtered class label for each of the bounding boxes (filtered_num_boxes).
        """
        # 1) Clipping boxes that go outside of frame
        bounding_boxes = clip_boxes_to_image(boxes=bounding_boxes, size=self.image_size)

        # 2) Remove small boxes
        keep = remove_small_boxes(boxes=bounding_boxes, min_size=16)
        bounding_boxes = bounding_boxes[keep]
        class_probits = class_probits[keep]
        labels = labels[keep]

        # 3) Remove boxes with highest probability of being a background
        keep = labels != 0
        bounding_boxes = bounding_boxes[keep]
        class_probits = class_probits[keep]
        labels = labels[keep]

        # 4) Removing boxes with small probability
        keep = class_probits > self.probability_threshold
        bounding_boxes = bounding_boxes[keep]
        class_probits = class_probits[keep]
        labels = labels[keep]

        # 5) Remove overlapping objects of same class using Non-max suppresion (NMS).
        keep = batched_nms(
            boxes=bounding_boxes,
            scores=class_probits,
            idxs=labels,
            iou_threshold=self.iou_threshold,
        )
        bounding_boxes = bounding_boxes[keep]
        class_probits = class_probits[keep]
        labels = labels[keep]

        return class_probits, bounding_boxes, labels

    def forward(self, fpn_outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        fpn_outputs = self.linker_layer(fpn_outputs=fpn_outputs)
        (
            all_anchors,  # (num_anchors, 4)
            all_objectness_logits,  # (num_anchors)
            all_anchor_deltas,  # (num_anchors, 4)
            filtered_objectness_logits,  # (num_proposals)
            filtered_proposals,  # (num_proposals, 4)
        ) = self.rpn(fpn_feature_map_outputs=fpn_outputs)

        filtered_pooled_proposals = self.roi(
            fpn_feature_map_outputs=fpn_outputs,
            proposals=filtered_proposals,
        )

        class_logits, filtered_proposals, proposal_deltas = self.output_heads(
            pooled_proposals_per_feature_map=filtered_pooled_proposals,
            proposals=filtered_proposals,
        )

        if self.training:
            distance_head_output = self.distance_head(filtered_pooled_proposals)
            size, yaw, keypoints = self.attribute_head(filtered_pooled_proposals)

            return {
                "rpn": (
                    all_anchors,  # (num_anchors, 4)
                    all_objectness_logits,  # (num_anchors)
                    all_anchor_deltas,  # (num_anchors, 4)
                    filtered_objectness_logits,  # (num_proposals)
                    filtered_proposals,  # (num_proposals, 4)
                ),
                "faster-rcnn": (class_logits, filtered_proposals, proposal_deltas),
                "mono-rcnn": (
                    distance_head_output,  # (num_proposals, 4)
                    size,  # (num_proposals, 3)
                    yaw,  # (num_proposals, 1)
                    keypoints,  # (num_proposals, 9)
                ),
            }
        class_probits, pred_bounding_boxes, labels = (
            self._fetch_probabilities_boxes_and_labels(
                proposals=filtered_proposals,
                bbox_regression_deltas=proposal_deltas,
                class_logits=class_logits,
            )
        )
        class_probits, pred_bounding_boxes, labels = self._filter_boxes(
            bounding_boxes=pred_bounding_boxes,
            class_probits=class_probits,
            labels=labels,
        )

        pooled_boxes = self.roi(
            fpn_feature_map_outputs=fpn_outputs,
            proposals=pred_bounding_boxes,
        )
        distance_head_output = self.distance_head(pooled_boxes)
        size, yaw, keypoints = self.attribute_head(pooled_boxes)

        return {
            "rpn": (filtered_proposals),
            "faster-rcnn": (
                class_probits,
                pred_bounding_boxes,
                labels,
            ),
            "mono-rcnn": (distance_head_output, size, yaw, keypoints),
        }
