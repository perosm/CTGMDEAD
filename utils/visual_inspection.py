import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import dataset.dataset_utils as KITTIUtils
from matplotlib import patches
from dataset.dataset_utils import TaskEnum


def plot_task_gt(task_groundtruth: dict[str, torch.Tensor]):
    tasks = task_groundtruth.keys()
    fig, ax = plt.subplots(len(tasks), 1, figsize=(8, 36))

    for i, (task, groundtruth) in enumerate(task_groundtruth.items()):
        if task == TaskEnum.object_detection_3d:
            img = (
                task_groundtruth[TaskEnum.input]
                .squeeze(0)
                .permute(1, 2, 0)
                .detach()
                .cpu()
                .numpy()
                .astype(np.int16)
            )
            groundtruth = groundtruth.squeeze(0).detach().cpu().numpy()
            for object_info in groundtruth:
                left, top, right, bottom = object_info[4:8].astype(np.int16)
                # changing start of axes
                # new_top = abs(KITTIUtils.NEW_H - top)
                # new_bottom = abs(KITTIUtils.NEW_H - bottom)
                width = left - right
                height = top - bottom
                print(object_info[4:8].astype(np.int16))
                #  rect = patches.Rectangle(
                #     xy=(left, KITTIUtils.KITTI_H - top - height),
                #     width=width,
                #     height=height,
                #     linewidth=2,
                #     edgecolor="r",
                #     facecolor="none",
                # )
                # ax[i].add_patch(rect)
                img[top:bottom, left, :] = (255, 0, 0)  # left line
                img[top:bottom, right, :] = (255, 0, 0)  # right line
                img[top, left:right, :] = (255, 0, 0)  # top line
                img[bottom, left:right, :] = (255, 0, 0)  # bottom line

        else:
            img = (
                groundtruth.squeeze(0)
                .permute(1, 2, 0)
                .detach()
                .cpu()
                .numpy()
                .astype(np.int16)
            )
        ax[i].imshow(img)
        ax[i].set_title(f"Task: {task}")
        ax[i].axis("off")

    def on_key(event):
        plt.close()

    plt.tight_layout()
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
