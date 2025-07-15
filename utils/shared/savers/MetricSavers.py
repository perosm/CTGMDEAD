import os
import abc
import pathlib
import yaml

from utils.shared.aggregators.MetricsAggregator import MetricsAggregator
from utils.shared.savers.Saver import Saver
from utils.shared.train_utils import DEVICE


class MetricsSaver(Saver):

    def __init__(
        self,
        metrics_aggregator: MetricsAggregator,
        save_dir: pathlib.Path,
        name: str,
        device: str = DEVICE,
    ):
        super().__init__(metrics_aggregator, save_dir)
        self.device = device
        self.name = name

    def save(self) -> None:
        per_task_metrics = self.aggregator.return_aggregated()
        with open(self.save_dir / f"{self.name}.yaml", "w") as f:
            yaml.dump(per_task_metrics, f)
