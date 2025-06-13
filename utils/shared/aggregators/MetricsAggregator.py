import torch
from torch import nn

from utils.shared.aggregators.Aggregator import Aggregator


class MetricsAggregator(Aggregator):

    def __init__(
        self,
        task_metrics: dict[str, list[nn.Module]],
        num_batches_total: int,
        device: str = "cpu",
    ):
        self.batch_cnt = 0
        self.num_batches = num_batches_total
        self.per_task_num_batches_count = {task: 0 for task in task_metrics.keys()}
        self.task_metrics_per_epochs: dict[str, dict[str, list[float]]] = {
            task: {metric.__class__.__name__: 0.0 for metric in task_metrics[task]}
            for task in task_metrics.keys()
        }

    def aggregate_per_batch(
        self, per_batch_values: dict[str, dict[str, torch.Tensor]]
    ) -> None:
        tasks_evaluated = {task: False for task in per_batch_values.keys()}
        for task, metrics in per_batch_values.items():
            for metric_name, metric_value in metrics.items():
                if tasks_evaluated[task] == False:
                    # If task is evaluated -> set flag to True and +1 for counter
                    tasks_evaluated[task] = True
                    self.per_task_num_batches_count[task] += 1
                self.task_metrics_per_epochs[task][metric_name] += metric_value.item()

        self.batch_cnt += 1

        if self.batch_cnt == self.num_batches:
            self._aggregate_per_epoch()

    def _aggregate_per_epoch(self) -> None:
        for task, metrics in self.task_metrics_per_epochs.items():
            for metric_name, _ in metrics.items():
                self.task_metrics_per_epochs[task][
                    metric_name
                ] = self.task_metrics_per_epochs[task][metric_name] / max(
                    self.per_task_num_batches_count[task], 1
                )  # TODO: remove, added just for purposes of debugging to prevent 0 division since we are using a subset of the samplelist

    def return_aggregated(self) -> dict[str, dict[str, torch.Tensor]]:
        return self.task_metrics_per_epochs
