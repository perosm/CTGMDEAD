import torch
from torch import nn
import torch.nn.functional as F
from collections import defaultdict

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
        # We do intersection of tasks because of co-training
        # since not all ground truth data is available for each image
        gt_tasks = set(gt.keys())
        pred_tasks = set(pred.keys())
        tasks = gt_tasks.intersection(pred_tasks)
        for task in tasks:
            if self.task_losses[task]:  # quick fix to autoencoder structure
                for loss in self.task_losses[task]:
                    per_task_losses[task][loss.__class__.__name__] = loss(
                        pred[task], gt[task]
                    )
                    total_loss += per_task_losses[task][loss.__class__.__name__]

        return total_loss, per_task_losses
