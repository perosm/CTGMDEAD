import torch
from torch import nn
import torch.nn.functional as F


class SSIM(nn.Module):
    def __init__(self):
        """
        https://arxiv.org/pdf/2006.13846
        https://drive.google.com/file/d/1rM3fMJ45F-bVYSz7dL8y1_mBDChIrs8g/view
        """
        super().__init__()

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        pass  # TODO: finish
