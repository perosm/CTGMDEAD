import torch
from torch import nn

"""
Pictures are evaluated with batch size 1.
"""
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MaskedAverageRelativeError(nn.Module):
    # used for picking N worst frames
    # lower MaskedAverageRelative = better -> False
    higher = False

    def __init__(self):
        self.eval()
        super().__init__()

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        mask = torch.where(gt != 0, 1, 0).to(DEVICE)
        valid_points = mask.sum()
        gt = torch.where(mask == 0, 1, gt)
        pred = torch.where(mask == 0, 1, pred)
        masked_avg_rel_error = (
            (torch.abs(pred - gt) / gt).sum((2, 3)) / valid_points
        ).mean()
        return masked_avg_rel_error.cpu().item()


class MaskedRMSE(nn.Module):
    # used for picking N worst frames
    # lower MaskedRMSE = better -> False
    higher = False

    def __init__(self):
        super().__init__()
        self.eval()

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        mask = torch.where(gt != 0, 1, 0).to(DEVICE)
        valid_points = mask.sum()
        masked_rmse = torch.sqrt(
            ((mask * pred - gt) ** 2).sum((2, 3)) / valid_points
        ).mean()

        return masked_rmse.cpu().item()


class MaskedThresholdAccracy(nn.Module):
    # used for picking N worst frames
    # higher MaskedThresholdAcuraccy = better -> True
    higher = True

    def __init__(self, threshold: float = 1.25):
        super().__init__()
        self.threshold = threshold
        self.eval()

    def forward(self, pred, gt) -> torch.Tensor:
        mask = torch.where(gt != 0, 1, 0).to(DEVICE)
        valid_points = mask.sum()
        gt = torch.where(mask == 0, 1, gt)
        pred = torch.where(mask == 0, 1, pred)
        masked_threshold_accuracy = (
            torch.max(pred / gt, gt / pred) < self.threshold
        ) * mask
        masked_threshold_accuracy = (
            masked_threshold_accuracy.float().sum() / valid_points
        )

        return masked_threshold_accuracy.cpu().item()


class MaskedMeanAbsoluteError(nn.Module):
    # used for picking N worst frames
    # lower MaskedMAE = better -> False
    higher = False

    def __init__(self):
        super().__init__()
        self.eval()

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        mask = torch.where(gt != 0, 1, 0).to(DEVICE)
        valid_points = mask.sum()
        masked_mae = (torch.abs((mask * pred - gt)).sum((2, 3)) / valid_points).mean()

        return masked_mae.cpu().item()
