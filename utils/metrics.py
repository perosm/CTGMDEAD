import torch
from torch import nn


#################### DEPTH ####################
"""
Pictures are evaluated with batch size 1.
"""
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MaskedAverageRelativeError(nn.Module):
    def __init__(self):
        super().__init__()
        # used for picking N worst frames
        # lower MaskedAverageRelative = better -> False
        self.higher = False

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
    def __init__(self):
        super().__init__()
        # used for picking N worst frames
        # lower MaskedRMSE = better -> False
        self.higher = False

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        mask = torch.where(gt != 0, 1, 0).to(DEVICE)
        valid_points = mask.sum()
        masked_rmse = torch.sqrt(
            ((mask * pred - gt) ** 2).sum((2, 3)) / valid_points
        ).mean()

        return masked_rmse.cpu().item()


class MaskedThresholdAccracy(nn.Module):
    def __init__(self, threshold: float = 1.25):
        super().__init__()
        # used for picking N worst frames
        # higher MaskedThresholdAcuraccy = better -> True
        self.higher = True
        self.threshold = threshold

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
    def __init__(self):
        super().__init__()
        # used for picking N worst frames
        # lower MaskedMAE = better -> False
        self.higher = False

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        mask = torch.where(gt != 0, 1, 0).to(DEVICE)
        valid_points = mask.sum()
        masked_mae = (torch.abs((mask * pred - gt)).sum((2, 3)) / valid_points).mean()

        return masked_mae.cpu().item()


#################### ROAD DETECTION ####################
class ConfusionMatrix(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self) -> torch.Tensor:
        pass
