import torch
from torch import nn


class IoU(nn.Module):
    higher = True

    def __init__(self):
        super().__init__()
        self.threshold = 0.5
        self.eval()

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        pred = (pred > self.threshold).bool()
        gt = gt.bool()
        intersection = (pred & gt).sum()
        union = (pred | gt).sum()

        return intersection / union


class Precision(nn.Module):
    higher = True

    def __init__(self):
        super().__init__()
        self.threshold = 0.5
        self.epsilon = 1e-5
        self.eval()

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        pred = (pred > self.threshold).bool()
        gt = gt.bool()
        tp = (pred & gt).sum()
        fp = (pred & ~gt).sum()

        return tp / (tp + fp + self.epsilon)


class Recall(nn.Module):
    higher = True

    def __init__(self):
        super().__init__()
        self.threshold = 0.5
        self.eval()

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        pred = (pred > self.threshold).bool()
        gt = gt.bool()
        tp = (pred & gt).sum()
        fn = (~pred & gt).sum()

        return tp / (tp + fn)


class FalsePositiveRate(nn.Module):
    higher = False

    def __init__(self):
        super().__init__()
        self.threshold = 0.5
        self.eval()

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        pred = (pred > self.threshold).bool()
        gt = gt.bool()
        fp = (pred & ~gt).sum()
        tn = (~pred & ~gt).sum()

        return fp / (fp + tn)


class TrueNegativeRate(nn.Module):
    higher = True

    def __init__(self):
        super().__init__()
        self.threshold = 0.5
        self.eval()

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        pred = (pred > self.threshold).bool()
        gt = gt.bool()
        fp = (pred & ~gt).sum()
        tn = (~pred & ~gt).sum()

        return tn / (tn + fp)


class F1Score(nn.Module):
    higher = True

    def __init__(self):
        super().__init__()
        self.threshold = 0.5
        self.epsilon = 1e-5
        self.eval()

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        pred = (pred > self.threshold).bool()
        gt = gt.bool()
        tp = (pred & gt).sum()
        fp = (pred & ~gt).sum()
        fn = (~pred & gt).sum()
        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)
        return 2 * precision * recall / (precision + recall)
