import torch
from torch import nn


class IoU(nn.Module):
    def __init__(self):
        super().__init__()
        self.threshold = 0.5

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        pred = (pred > self.threshold).to(torch.int8)
        gt = gt.to(torch.int8)
        intersection = torch.sum((pred & gt) == 1)
        union = torch.sum((pred | gt) == 1)

        return intersection / union


class Precision(nn.Module):
    def __init__(self):
        super().__init__()
        self.threshold = 0.5

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        pred = (pred > self.threshold).to(torch.int8)
        gt = gt.to(torch.int8)
        tp = torch.sum((pred & gt) == 1)
        tn = torch.sum((~pred & ~gt) == 1)

        return tp / (tp + tn)


class Recall(nn.Module):
    def __init__(self):
        super().__init__()
        self.threshold = 0.5

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        pred = (pred > self.threshold).to(torch.int8)
        gt = gt.to(torch.int8)
        tp = torch.sum((pred & gt) == 1)
        fn = torch.sum((~pred & gt) == 1)

        return tp / (tp + fn)


class FalsePositiveRate(nn.Module):
    def __init__(self):
        super().__init__()
        self.threshold = 0.5

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        pred = (pred > self.threshold).to(torch.int8)
        gt = gt.to(torch.int8)
        fp = torch.sum((pred & ~gt) == 1)
        tn = torch.sum((~pred & ~gt) == 1)

        return fp / (fp + tn)


class TrueNegativeRate(nn.Module):
    def __init__(self):
        super().__init__()
        self.threshold = 0.5

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        pred = (pred > self.threshold).to(torch.int8)
        gt = gt.to(torch.int8)
        fp = torch.sum((pred & ~gt) == 1)
        tn = torch.sum((~pred & ~gt) == 1)

        return tn / (tn + fp)
