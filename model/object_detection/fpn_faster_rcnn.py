import torch
from torch import nn
from model.object_detection.rpn import RegionProposalNetwork
from model.object_detection.roi import ROINetwork
from utils.shared.dict_utils import list_of_dict_to_dict


class FPNFasterRCNN(nn.Module):
    def __init__(self, configs: dict):
        super().__init__()
        self.image_size = configs["image_size"]
        self.rpn = self._configure_region_proposal_network(
            rpn_config=configs["rpn"]
        )  # TODO: add per feature map RegionProposalNetwork?
        self.roi = self._configure_roi_network(roi_config=configs["roi_network"])

    def _configure_region_proposal_network(
        self, rpn_config: list[dict]
    ) -> RegionProposalNetwork:
        rpn_config.append({"image_size": self.image_size})
        rpn_config = list_of_dict_to_dict(
            list_of_dicts=rpn_config, new_dict={}, depth_cnt=1
        )
        return RegionProposalNetwork(configs=rpn_config)

    def _configure_roi_network(self, roi_config: list[dict]) -> ROINetwork:
        roi_config = list_of_dict_to_dict(
            list_of_dicts=roi_config, new_dict={}, depth_cnt=1
        )
        pool_output_size = roi_config["pool_output_size"]
        return ROINetwork(image_size=self.image_size, pool_output_size=pool_output_size)

    def forward(
        self,
        encoder_outputs: tuple[torch.Tensor, ...],
        decoder_outputs: tuple[torch.Tensor, ...],
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        fpn_outputs = {}
        for i in range(len(encoder_outputs)):
            fpn_outputs[f"fpn_{i}"] = (
                encoder_outputs[i]
                + decoder_outputs[
                    len(encoder_outputs) - i - 1
                ]  # TODO: needs a bit of change :)
            )

        if self.training:
            objectness_score_per_feature_map, proposals_per_feature_map = self.rpn(
                fpn_outputs, y_true
            )
        else:
            objectness_score_per_feature_map, proposals_per_feature_map = self.rpn(
                fpn_outputs
            )

        class_logits, bounding_box_deltas = self.roi(proposals_per_feature_map)

        return class_logits, bounding_box_deltas
