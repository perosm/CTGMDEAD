import torch
import numpy as np
from torch import nn
from utils.shared.aggregators.Aggregator import Aggregator


class LossAggregator(Aggregator):
    def __init__(
        self,
        task_losses: dict[str, list[nn.Module]],
        epochs: int,
        num_batches_total: int,
        device: str,
    ):
        """
        Used to track total loss per epoch and loss per task per epoch.
        """
        self.epoch_cnt = 0
        self.epochs = epochs
        self.num_batches_total = num_batches_total
        self.per_task_num_batches_count = {task: 0 for task in task_losses}
        self.total_loss_per_epochs = np.zeros(epochs)
        self.task_losses_per_epochs: dict[str, dict[str, list[float]]] = {
            task: {
                loss.__class__.__name__: np.zeros(epochs) for loss in task_losses[task]
            }
            for task in task_losses.keys()
        }

    def aggregate_per_batch(
        self, per_batch_values: dict[str, dict[str, torch.Tensor]]
    ) -> None:
        # task which are trained in this batch
        tasks_trained = per_batch_values.keys()
        for task, losses in per_batch_values.items():
            for loss_name, loss_value in losses.items():
                self.task_losses_per_epochs[task][loss_name][self.epoch_cnt] += (
                    loss_value.detach().cpu().numpy()
                )

        # Update batch count per task
        total_batch_cnt = 0
        for task in tasks_trained:
            self.per_task_num_batches_count[task] += 1
            total_batch_cnt += 1

        if total_batch_cnt == self.num_batches_total:
            self._aggregate_per_epoch()

    def _aggregate_per_epoch(self) -> None:
        for task, losses in self.task_losses_per_epochs.items():
            for _, loss_value in losses.items():
                loss_value[self.epoch_cnt] = (
                    loss_value[self.epoch_cnt] / self.per_task_num_batches_count[task]
                )
                self.total_loss_per_epochs[self.epoch_cnt] += loss_value[self.epoch_cnt]

        # New epoch -> Reset batch count per task
        self.epoch_cnt += 1
        for task in self.per_task_num_batches_count:
            self.per_task_num_batches_count[task] = 0

    def return_aggregated(self) -> dict[str, dict[str, torch.Tensor]]:
        return self.task_losses_per_epochs
