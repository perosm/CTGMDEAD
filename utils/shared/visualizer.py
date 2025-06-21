import abc
import pathlib

import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.shared.enums import TaskEnum


class VisualizerStrategy(abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def visualize(self, pred: np.ndarray, gt: np.ndarray, image: np.ndarray):
        pass


class Visualizer:
    def __init__(
        self,
        visualizers: dict[str, VisualizerStrategy],
        save_dir: pathlib.Path,
        epoch: int,
    ):
        self.visualizers = visualizers
        self.save_dir = save_dir
        self.epoch = epoch
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot_visualizations(
        self,
        pred: dict[str, torch.Tensor],
        gt: dict[str, torch.Tensor],
        image: torch.Tensor,
    ):
        image = image * 255.0
        data_to_plot = {
            TaskEnum.input: image.squeeze(0)
            .cpu()
            .permute(1, 2, 0)
            .numpy()
            .astype(np.uint8)
        }
        pred_tasks = set(pred.keys())
        gt_tasks = set(gt.keys())
        tasks = pred_tasks.intersection(gt_tasks)
        for task in tasks:
            data_to_plot.update(
                {
                    **self.visualizers[task].visualize(
                        pred[task].cpu(),
                        gt[task],
                        image.squeeze(0).cpu().clone(),
                    )
                }
            )

        self._plot(data_to_plot)

    def _plot(self, data_to_plot: dict[str, np.ndarray]):
        save_path = self.save_dir / f"per_task_pred_and_gt_{self.epoch}.png"
        num_subplots = len(data_to_plot)

        fig, axes = plt.subplots(num_subplots, 1, figsize=(16, 10))

        if num_subplots == 1:
            axes = [axes]

        for ax, (task, plot_data) in zip(axes, data_to_plot.items()):
            ax.imshow(plot_data)
            ax.set_title(task)
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
