import torch
from torch import nn
import torch.nn.functional as F
from collections import defaultdict


class MultiTaskLoss(nn.Module):
    def __init__(self, task_losses: dict[str, list[nn.Module]]):
        super().__init__()
        self.task_losses = task_losses

    def forward(
        self, pred: dict[str, torch.Tensor], gt: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        total_loss = 0.0
        per_task_losses = {
            task: defaultdict(torch.Tensor) for task in self.task_losses.keys()
        }  # used to keep track of losses per task
        for task, losses in self.task_losses.items():
            for loss in losses:
                per_task_losses[task][loss.__class__.__name__] = loss(
                    pred[task], gt[task]
                )
                total_loss += per_task_losses[task][loss.__class__.__name__]

        return total_loss, per_task_losses


class MaskedMAE(nn.Module):
    def __init__(self, name: str = "MaskedMAE", device: str = "cuda"):
        super().__init__()
        self.name = name
        self._l1_loss = nn.L1Loss(reduction="none")
        self.device = device

    def forward(self, pred, gt):
        mask = torch.where(gt != 0, 1, 0).to(self.device)
        valid_points = mask.sum()
        loss = self._l1_loss(pred * mask, gt).sum((2, 3)) / valid_points
        return loss.mean()


class GradLoss(nn.Module):
    def __init__(self, name: str = "GradLoss", device: str = "cuda"):
        super().__init__()
        self.name = name
        sobel_x = (
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        ).to(device)
        sobel_y = (
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        ).to(device)

        self.conv_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False).to(device)
        self.conv_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False).to(device)

        self.conv_x.weight = nn.Parameter(sobel_x, requires_grad=False)
        self.conv_y.weight = nn.Parameter(sobel_y, requires_grad=False)

    def forward(self, pred, gt) -> torch.Tensor:
        grad_pred_x = self.conv_x(pred)
        grad_pred_y = self.conv_y(pred)
        grad_gt_x = self.conv_x(gt)
        grad_gt_y = self.conv_y(gt)

        grad_pred = torch.sqrt(grad_pred_x**2 + grad_pred_y**2)
        grad_gt = torch.sqrt(grad_gt_x**2 + grad_gt_y**2)

        loss = F.l1_loss(grad_pred, grad_gt)

        return loss


class BinaryCrossEntropyLoss(nn.Module):

    def __init__(self, name: str = "BinaryCrossEntropyLoss"):
        super().__init__()
        self.name = name
        self.loss = nn.BCELoss(reduction="mean")

    def forward(self, pred, gt):
        return self.loss(pred, gt)
