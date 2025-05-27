import torch
from torch import nn


class PredictionPostprocessor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, prediction: torch.Tensor, projection_matrix: torch.Tensor
    ) -> torch.Tensor:

        return prediction  # TODO: Should there be any postprocessing ?
