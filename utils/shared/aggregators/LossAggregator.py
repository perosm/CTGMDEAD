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
    ):
        """
        Used to track total loss per epoch and loss per task per epoch.
        """
        self.epoch_cnt = 0
        self.epochs = epochs
        self.batch_cnt = 0
        self.num_batches_total = num_batches_total
        self.per_task_num_batches_count = {task: 0 for task in task_losses.keys()}
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
        tasks_trained = {task: False for task in per_batch_values.keys()}
        for task, losses in per_batch_values.items():
            for loss_name, loss_value in losses.items():
                if tasks_trained[task] == False:
                    # If task is trained -> set flag to True and +1 for counter
                    tasks_trained[task] = True
                    self.per_task_num_batches_count[task] += 1
                self.task_losses_per_epochs[task][loss_name][self.epoch_cnt] += (
                    loss_value.detach().cpu().numpy()
                )

        self.batch_cnt += 1

        if self.batch_cnt == self.num_batches_total:
            self._aggregate_per_epoch()

    def _aggregate_per_epoch(self) -> None:
        for task, losses in self.task_losses_per_epochs.items():
            for _, loss_value in losses.items():
                loss_value[self.epoch_cnt] = loss_value[self.epoch_cnt] / max(
                    self.per_task_num_batches_count[task], 1
                )  # TODO: remove, added just for purposes of debugging to prevent 0 division since we are using a subset of the samplelist
                self.total_loss_per_epochs[self.epoch_cnt] += loss_value[self.epoch_cnt]

        # New epoch -> Reset batch count and batch count per task
        self.batch_cnt = 0
        for task in self.per_task_num_batches_count:
            self.per_task_num_batches_count[task] = 0
        self.epoch_cnt += 1

    def return_aggregated(self) -> dict[str, dict[str, torch.Tensor]]:
        return self.task_losses_per_epochs
