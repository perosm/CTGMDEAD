import torch
from torch import nn
from model.object_detection.rpn import RegionProposalNetwork
from model.object_detection.roi import ROINetwork
from model.object_detection.output_heads import OutputHeads
from utils.shared.dict_utils import list_of_dict_to_dict


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
        self.training = configs["training"]
        self.image_size = configs["image_size"]
        self.pool_output_size = configs["pool_output_size"]
        self.num_channels_per_feature_map = configs["num_channels_per_feature_map"]
        self.out_channels = configs["out_channels"]
        self.linker_layer = self._configure_linker_layer(
            num_channels_per_feature_map=self.num_channels_per_feature_map,
            out_channels=self.out_channels,
        )
        # TODO: add per feature map RegionProposalNetwork?
        self.rpn = self._configure_region_proposal_network(rpn_config=configs["rpn"])
        self.roi = self._configure_region_of_interest_network(roi_config=configs["roi"])
        self.output_heads = self._configure_output_heads(
            output_heads_config=configs["output_heads"]
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
        rpn_config.append({"training": self.training})
        rpn_config = list_of_dict_to_dict(
            list_of_dicts=rpn_config, new_dict={}, depth_cnt=1
        )
        return RegionProposalNetwork(configs=rpn_config)

    def _configure_region_of_interest_network(
        self, roi_config: list[dict]
    ) -> ROINetwork:
        roi_config = list_of_dict_to_dict(roi_config, new_dict={}, depth_cnt=1)
        return ROINetwork(
            image_size=self.image_size,
            pool_output_size=self.pool_output_size,
            feature_map_names=roi_config["feature_map_names"],
            sampling_ratio=roi_config["sampling_ratio"],
        )

    def _configure_output_heads(self, output_heads_config: list[dict]) -> OutputHeads:
        output_heads_config = list_of_dict_to_dict(
            list_of_dicts=output_heads_config,
        )
        return OutputHeads(
            training=self.training,
            image_size=self.image_size,
            pool_size=self.pool_output_size,
            num_classes=output_heads_config["num_classes"],
            score_threshold=output_heads_config["score_threshold"],
            iou_threshold=output_heads_config["iou_threshold"],
            top_k_boxes_training=output_heads_config["top_k_boxes_training"],
            top_k_boxes_testing=output_heads_config["top_k_boxes_testing"],
        )

    def forward(self, fpn_outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        fpn_outputs = self.linker_layer(fpn_outputs=fpn_outputs)
        objectness_scores, proposals = self.rpn(fpn_feature_map_outputs=fpn_outputs)

        pooled_proposals = self.roi(
            fpn_feature_map_outputs=fpn_outputs,
            proposals=proposals,
        )

        class_probits, bounding_boxes = self.output_heads(pooled_proposals, proposals)
        if self.training:
            return {
                "rpn": (objectness_scores, proposals),
                "faster-rcnn": (class_probits, bounding_boxes),
            }

        return class_probits, bounding_boxes
