import pathlib
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
from collections import defaultdict

from utils.shared.aggregators.LossAggregator import LossAggregator
from utils.shared.savers.Saver import Saver


class LossSaver(Saver):
    def __init__(
        self, loss_aggregator: LossAggregator, save_dir: pathlib.Path, device="cuda"
    ):
        super().__init__(loss_aggregator, save_dir)
        self.device = device

    def save(self) -> None:
        task_loss_per_epoch = self.aggregator.return_aggregated()
        task_loss_per_epoch = LossSaver._make_it_json_serializable(task_loss_per_epoch)
        with open(self.save_dir / "losses.json", "w") as f:
            json.dump(task_loss_per_epoch, f)

    @staticmethod
    def _make_it_json_serializable(
        task_loss_per_epoch: dict[str, dict[str, torch.Tensor]],
    ) -> dict[str, dict[str, list[float]]]:
        task_loss_per_epoch_serializable = defaultdict(dict[str, list])
        for task in task_loss_per_epoch.keys():
            for loss in task_loss_per_epoch[task]:
                task_loss_per_epoch_serializable[task][loss] = task_loss_per_epoch[
                    task
                ][loss].tolist()

        return task_loss_per_epoch_serializable

    def save_plot(self) -> None:  # TODO: should I make one plot for each of the tasks?
        task_loss_per_epoch = self.aggregator.return_aggregated()
        epochs_array = np.arange(0, self.aggregator.epochs)
        total_loss = torch.zeros(self.aggregator.epochs).to(self.device)
        fig, ax = plt.subplots(len(task_loss_per_epoch.keys()) + 1, 1, figsize=(16, 10))
        for row, (task, losses) in enumerate(task_loss_per_epoch.items()):
            total_loss_per_task = torch.zeros(self.aggregator.epochs).to(self.device)
            for loss_name, loss_value in losses.items():
                ax[row + 1].plot(
                    epochs_array,
                    loss_value.cpu().numpy(),
                    label=loss_name,
                )
                ax[row + 1].set_title(task)
                ax[row + 1].legend()
                total_loss_per_task += loss_value
            total_loss += total_loss_per_task
            ax[0].plot(epochs_array, total_loss_per_task.cpu().numpy(), label=task)
        ax[0].plot(epochs_array, total_loss.cpu().numpy())
        ax[0].set_title("Losses")
        ax[0].legend()
        plt.ioff()
        plt.savefig(self.save_dir / "losses.png")
        plt.close()
