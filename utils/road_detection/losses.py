import torch
from torch import nn


class BinaryCrossEntropyLoss(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.scale_factor = kwargs.get("scale_factor", 1.0)
        self.loss = nn.BCELoss(reduction="mean")

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        return self.scale_factor * self.loss(pred, gt)
