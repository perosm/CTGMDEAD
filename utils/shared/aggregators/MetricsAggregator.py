import torch
from torch import nn

from utils.shared.aggregators.Aggregator import Aggregator


class MetricsAggregator(Aggregator):
    def __init__(
        self,
        task_metrics: dict[str, list[nn.Module]],
        num_batches: int,
        device: str = "cpu",
    ):
        self.batch_cnt = 0
        self.num_batches = num_batches
        self.task_metrics_per_epochs: dict[str, dict[str, list[float]]] = {
            task: {metric.__class__.__name__: 0.0 for metric in task_metrics[task]}
            for task in task_metrics.keys()
        }

    def aggregate_per_batch(
        self, per_batch_values: dict[str, dict[str, torch.Tensor]]
    ) -> None:
        self.batch_cnt += 1
        for task, metrics in per_batch_values.items():
            for metric_name, metric_value in metrics.items():
                self.task_metrics_per_epochs[task][metric_name] += metric_value.item()

        if self.batch_cnt == self.num_batches:
            self._aggregate_per_epoch()

    def _aggregate_per_epoch(self) -> None:
        for task, metrics in self.task_metrics_per_epochs.items():
            for metric_name, _ in metrics.items():
                self.task_metrics_per_epochs[task][metric_name] = (
                    self.task_metrics_per_epochs[task][metric_name] / self.num_batches
                )

    def return_aggregated(self) -> dict[str, dict[str, torch.Tensor]]:
        return self.task_metrics_per_epochs
