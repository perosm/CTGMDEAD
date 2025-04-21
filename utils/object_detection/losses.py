import torch
from torch import nn


class ClassificationLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.BCELoss(reduction="mean")

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.loss(y_pred, y_true)


class RegressionLoss(nn.Module):
    def __init__(self, regularization_factor: float = 10.0) -> None:
        super().__init__()
        self.regularization_factor = regularization_factor
        self.loss = nn.SmoothL1Loss(reduction="mean")

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.regularization_factor * self.loss(y_pred, y_true)
