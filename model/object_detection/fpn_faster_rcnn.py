import torch
from torch import nn
from model.object_detection.rpn import RegionProposalNetwork
from model.object_detection.roi import ROINetwork
from model.object_detection.anchor_generator import AnchorGenerator


class FPNFasterRCNN(nn.Module):
    def __init__(self, num_channels_per_feature_map: list[int]):
        super().__init__()
        self.rpn = RegionProposalNetwork(
            num_channels_per_feature_map
        )  # TODO: add per feature map RegionProposalNetwork?
        self.anchor_generator = AnchorGenerator(
            sizes=(  # TODO: which anchor sizes to choose?
                (512),
                (256),
                (128),
                (64),  # (32)
            ),
            aspect_ratios=(
                (0.5, 1.0, 2.0),
                (0.5, 1.0, 2.0),
                (0.5, 1.0, 2.0),
                (0.5, 1.0, 2.0),
            ),
        )
        self.roi = ROINetwork()

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

        objectness_score_per_feature_map, bbox_regression_per_feature_map = self.rpn(
            fpn_outputs
        )
