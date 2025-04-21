import pathlib
import numpy as np
import torch
import matplotlib.pyplot as plt
import json

from utils.aggregators.LossAggregator import LossAggregator
from utils.savers.Saver import Saver


class LossSaver(Saver):
    def __init__(
        self, loss_aggregator: LossAggregator, save_dir: pathlib.Path, device="cuda"
    ):
        super().__init__(loss_aggregator, save_dir)
        self.device = device

    def save(self) -> None:
        task_loss_per_epoch = self.aggregator.return_aggregated()
        with open(self.save_dir / "losses.json") as f:
            json.dump(task_loss_per_epoch, f)

    def save_plot(self) -> None:  # TODO: should I make one plot for each of the tasks?
        task_loss_per_epoch = self.aggregator.return_aggregated()
        epochs_array = np.arange(0, self.aggregator.epochs)
        total_loss = torch.zeros(self.aggregator.epochs).to(self.device)
        fig, ax = plt.subplots(len(task_loss_per_epoch.keys()) + 1, 1)
        for row, (task, losses) in enumerate(task_loss_per_epoch.items()):
            total_loss_per_task = torch.zeros(self.aggregator.epochs).to(self.device)
            for loss_name, loss_value in losses.items():
                ax[row + 1].plot(
                    epochs_array,
                    loss_value.cpu().numpy(),
                    label=loss_name,
                )
                ax[row + 1].set_title(task)
                total_loss_per_task += loss_value
            total_loss += total_loss_per_task
            ax[0].plot(epochs_array, total_loss_per_task.cpu().numpy(), label=task)
        ax[0].plot(epochs_array, total_loss.cpu().numpy())
        ax[0].set_title("Losses")
        fig.legend()

        plt.savefig(self.save_dir / "losses.png")
