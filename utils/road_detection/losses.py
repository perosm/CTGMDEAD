import torch
from torch import nn


class BinaryCrossEntropyLoss(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.lambda_factor = kwargs.get("lambda_factor", 1.0)
        self.loss = nn.BCELoss(reduction="mean")

    def forward(self, pred, gt) -> torch.Tensor:
        return self.loss(pred, gt)
