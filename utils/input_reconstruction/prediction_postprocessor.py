import torch
from torch import nn


class PredictionPostprocessor(nn.Module):
    def __init__(self, min_depth: float = 0.0, max_depth: float = 90.0):
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth

    def forward(
        self, prediction: torch.Tensor, projection_matrix: torch.Tensor
    ) -> torch.Tensor:
        return prediction
