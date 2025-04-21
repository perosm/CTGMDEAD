import torch
from torch import nn
from model.object_detection.anchor_generator import AnchorGenerator


class ROINetwork(nn.Module):
    def __init__(self, anchor_generator: AnchorGenerator):
        super().__init__()

    def forward(self) -> torch.Tensor:
        pass
