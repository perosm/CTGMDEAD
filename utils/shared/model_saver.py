import yaml
import pathlib
import torch
from torch import nn

from utils.shared.metrics import MultiTaskMetrics


class ModelSaver:

    def __init__(self, save_dir: pathlib.Path, task_metrics: MultiTaskMetrics):
        """
        Initializes the ModelSaver.

        Args:
            save_dir: Directory to save the models.

        """
        self.save_dir = save_dir
        self.task_metrics_higher_lower = {
            task: {
                metric.__class__.__name__: metric.higher
                for metric in task_metrics[task]
            }
            for task in task_metrics.keys()
        }
        self.best_task_metrics = None
        self.save_dir.mkdir(exist_ok=True, parents=True)

    def __call__(
        self, model: nn.Module, task_metrics: dict[str, dict[str, list[float]]]
    ):
        if self.best_task_metrics is None:
            self.best_task_metrics = task_metrics
            self._save_model(model, self.save_dir / "best_model.pth")
            self._save_best_metrics()

        cnt = 0
        tasks = task_metrics.keys()
        for task in tasks:
            for metric_name in task_metrics[task].keys():
                higher_lower_flag = self.task_metrics_higher_lower[task][metric_name]
                metric_value = task_metrics[task][metric_name]
                if higher_lower_flag is True:
                    # if higher_lower_flag is True the higher the metric value the better it is
                    cnt += (
                        1
                        if metric_value > self.best_task_metrics[task][metric_name]
                        else -1
                    )
                elif higher_lower_flag is False:
                    # if higher_lower_flag is False the lower the metric value the better it is
                    cnt += (
                        1
                        if metric_value < self.best_task_metrics[task][metric_name]
                        else -1
                    )

        if cnt >= 0:
            self._save_model(model, self.save_dir / "best_model.pth")
            self._save_best_metrics()

    def _save_model(self, model: nn.Module, path: str):
        torch.save(model.state_dict(), path)
        print(f"Saved model to {path}")

    def _save_best_metrics(self) -> None:
        yaml_filepath = self.save_dir / "eval" / "best_metrics.yaml"
        with open(yaml_filepath, "w") as file:
            yaml.dump(self.best_task_metrics, file)
