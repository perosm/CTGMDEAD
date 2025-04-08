import numpy as np
import torch
import matplotlib.pyplot as plt


def plot_task_gt(task_groundtruth: dict[str, torch.Tensor]):
    tasks = task_groundtruth.keys()
    fig, ax = plt.subplots(len(tasks), 1, figsize=(10, 50))

    for i, (task, groundtruth) in enumerate(task_groundtruth.items()):
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
