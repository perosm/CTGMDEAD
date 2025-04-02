import numpy as np
import torch
import matplotlib.pyplot as plt


def plot_task_gt(task_groundtruth: dict[str, torch.Tensor]):
    tasks = task_groundtruth.keys()
    fig, ax = plt.subplots(len(tasks), 1)

    for i, (task, groundtruth) in enumerate(task_groundtruth.items()):
        img = groundtruth.squeeze(0).permute(2, 1, 0).detach().cpu().numpy()
        # color = "rgb" if img.shape[-1] == 3 else "gray"
        ax[i].imshow(img)
        ax[i].set_title(f"Task: {task}")
        ax[i].axis("off")

    plt.tight_layout()
    plt.show()
