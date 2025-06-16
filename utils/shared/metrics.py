from collections import defaultdict

import torch

from torch import nn


class MultiTaskMetrics(nn.Module):
    def __init__(self, task_metrics: dict[str, list[nn.Module]]):
        super().__init__()
        self.task_metrics = task_metrics
        self.eval()

    def forward(
        self, pred: dict[str, torch.Tensor], gt: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        per_task_metrics = {
            task: defaultdict(torch.Tensor) for task in self.task_metrics.keys()
        }
        # We do intersection of tasks because of co-training
        # since not all ground truth data is available for each image
        gt_tasks = set(gt.keys())
        pred_tasks = set(pred.keys())
        tasks = gt_tasks.intersection(pred_tasks)
        for task in tasks:
            if self.task_metrics.get(task, None):  # quick fix to autoencoder structure
                for metric in self.task_metrics[task]:
                    per_task_metrics[task][metric.__class__.__name__] = metric(
                        pred[task], gt[task]
                    )

        return per_task_metrics
