import torch
import numpy as np

from utils.shared.visualizer import VisualizerStrategy
from utils.shared.enums import TaskEnum


class Visualizer(VisualizerStrategy):
    task = TaskEnum.input

    def __init__(self):
        pass

    def visualize(
        self, pred: torch.Tensor, gt: torch.Tensor, image: torch.Tensor
    ) -> dict[str, np.ndarray]:

        return {
            f"{self.task}_reconstructed": pred.squeeze()
            .permute(1, 2, 0)
            .numpy()
            .astype(np.float32)
        }
