import torch
import numpy as np

from utils.shared.visualizer import VisualizerStrategy
from utils.shared.enums import TaskEnum


class Visualizer(VisualizerStrategy):
    task = TaskEnum.road_detection

    def __init__(self):
        pass

    def visualize(
        self, pred: torch.Tensor, gt: torch.Tensor, image: torch.Tensor
    ) -> dict[str, np.ndarray]:

        gt = gt.cpu().squeeze().numpy()
        pred = pred.squeeze().numpy()
        image = image.numpy()

        pred_mask = pred > 0.5
        gt_mask = gt == 1
        tp = pred_mask & gt_mask
        # fp = pred_mask & ~gt_mask
        fn = ~pred_mask & gt_mask
        # tn = ~pred_mask & ~gt_mask

        light_green_overlay = np.array([0, 128, 0], dtype=image.dtype).reshape(3, 1)
        light_red_overlay = np.array([128, 0, 0], dtype=image.dtype).reshape(3, 1)

        image[:, tp] = light_green_overlay
        # image[:, fp] = light_red_overlay
        image[:, fn] = light_red_overlay
        # image[:, tn] = light_red_overlay

        return {self.task: np.transpose(image, (1, 2, 0)).astype(np.uint8)}
