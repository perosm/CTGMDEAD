import os
import abc
import pathlib
import json

from utils.shared.aggregators.MetricsAggregator import MetricsAggregator
from utils.shared.savers.Saver import Saver


class MetricsSaver(Saver):

    def __init__(
        self,
        metrics_aggregator: MetricsAggregator,
        save_dir: pathlib.Path,
        name: str,
        device: str = "cuda",
    ):
        super().__init__(metrics_aggregator, save_dir)
        self.device = device
        self.name = name

    def save(self) -> None:
        per_task_metrics = self.aggregator.return_aggregated()
        with open(self.save_dir / f"{self.name}.json") as f:
            json.dump(per_task_metrics, f)
