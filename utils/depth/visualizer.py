import torch
import numpy as np

from utils.shared.visualizer import VisualizerStrategy
from utils.shared.enums import TaskEnum


class Visualizer(VisualizerStrategy):
    task = TaskEnum.depth

    def __init__(self):
        pass

    def visualize(
        self, pred: torch.Tensor, gt: torch.Tensor, image: torch.Tensor
    ) -> dict[str, np.ndarray]:

        return {self.task: pred.permute(1, 2, 0).numpy().squeeze().astype(np.uint8)}
