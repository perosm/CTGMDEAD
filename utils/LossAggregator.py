import torch
from torch import nn
from collections import defaultdict


class LossAggregator:
    def __init__(self, task_losses: dict[str, list[nn.Module]]):
        """
        Used to track total loss per epoch and loss per task per epoch.
        Also used for visualizing training losses.
        """
        loss_per_epochs: list[float] = []
        task_losses_per_epochs: dict[str, dict[str, list[float]]] = {
            task: {loss.__class__.__name__: [] for loss in task_losses[task]}
            for task in task_losses.keys()
        }

    def aggregate_per_epoch(
        self, task_epoch_losses: dict[str, list[torch.Tensor]]
    ) -> None:
        for task in task_epoch_losses.keys():
            pass

    def aggregate_per_batch(self) -> None:
        pass

    def fetch_task_epochs_losses(self) -> dict[str, list[torch.Tensor]]:
        pass
