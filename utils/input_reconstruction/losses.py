import torch
from torch import nn
import torch.nn.functional as F


class MSE(nn.Module):
    higher = False

    def __init__(self, **kwargs):
        super().__init__()
        self.scale_factor = kwargs.get("scale_factor", 1.0)

    def forward(self, pred: torch.Tensor, gt: torch.Tensor):
        return self.scale_factor * F.mse_loss(pred, gt, reduction="mean")
