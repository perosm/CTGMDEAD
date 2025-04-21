import torch
from torch import nn
from collections import defaultdict
from utils.shared.aggregators.Aggregator import Aggregator


class LossAggregator(Aggregator):
    def __init__(
        self,
        task_losses: dict[str, list[nn.Module]],
        epochs: int,
        num_batches: int,
        device: str = "cuda",
    ):
        """
        Used to track total loss per epoch and loss per task per epoch.
        """
        self.epoch_cnt = 0
        self.epochs = epochs
        self.batch_cnt = 0
        self.num_batches = num_batches
        self.loss_per_epochs = torch.zeros(epochs).to(device)
        self.task_losses_per_epochs: dict[str, dict[str, list[float]]] = {
            task: {
                loss.__class__.__name__: torch.zeros(epochs).to(device)
                for loss in task_losses[task]
            }
            for task in task_losses.keys()
        }

    def aggregate_per_batch(
        self, per_batch_values: dict[str, dict[str, torch.Tensor]]
    ) -> None:
        self.batch_cnt += 1
        for task, losses in per_batch_values.items():
            for loss_name, loss_value in losses.items():
                self.task_losses_per_epochs[task][loss_name][
                    self.epoch_cnt
                ] += loss_value.detach()

        if self.batch_cnt == self.num_batches:
            self._aggregate_per_epoch()

    def _aggregate_per_epoch(self) -> None:
        for task, losses in self.task_losses_per_epochs.items():
            for _, loss_value in losses.items():
                self.loss_per_epochs[self.epoch_cnt] += (
                    loss_value[self.epoch_cnt] / self.num_batches
                )

        self.epoch_cnt += 1
        self.batch_cnt = 0

    def return_aggregated(self) -> dict[str, dict[str, torch.Tensor]]:
        return self.task_losses_per_epochs
