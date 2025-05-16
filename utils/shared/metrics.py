from collections import defaultdict

import torch

from torch import nn


class MultiTaskMetrics(nn.Module):
    def __init__(self, task_metrics: dict[str, list[nn.Module]]):
        super().__init__()
        self.task_metrics = task_metrics

    def forward(
        self, pred: dict[str, torch.Tensor], gt: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        per_task_metrics = {
            task: defaultdict(torch.Tensor) for task in self.task_metrics.keys()
        }

        for task, metrics in self.task_metrics.items():
            for metric in metrics:
                per_task_metrics[task][metrics.__class__.__name__] = metric(
                    pred[task], gt[task]
                )

        return per_task_metrics
