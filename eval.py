import os
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm

from dataset.kitti.KittiDataset import KittiDataset
from model.resnet import ResNet18
from model.decoder import UnetDecoder
from model.encoder_decoder import DepthEncoderDecoder
from utils.metrics import (
    MaskedAverageRelativeError,
    MaskedRMSE,
    MaskedThresholdAccracy,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEPTH_MAX = 90.0


def plot_input_and_depth(image, depth, pred, save_path, metric, worst_idx):
    image_np = image.cpu().numpy().squeeze(0).transpose(1, 2, 0) / 256.0
    depth_np = depth.cpu().numpy().squeeze(0).transpose(1, 2, 0)
    pred_np = pred.cpu().numpy().squeeze(0).transpose(1, 2, 0)
    mask = np.where(depth_np != 0, 1, 0)
    diff_np = mask * pred_np - depth_np

    vmin = min(depth_np.min(), pred_np.min())
    vmax = max(depth_np.max(), pred_np.max())
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(36, 24),
        subplot_kw={"aspect": "equal"},
    )
    fig.subplots_adjust(top=0.95, bottom=0.05, left=0.1, right=0.9, hspace=0.4)

    im0 = axes[0].imshow(image_np)
    axes[0].axis("off")

    im1 = axes[1].imshow(depth_np, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].axis("off")

    im2 = axes[2].imshow(pred_np, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[2].axis("off")
    plt.colorbar(im1, ax=[axes[1], axes[2]], pad=0.01)

    dmin = np.min(diff_np)
    dmax = np.max(diff_np)
    norm = mcolors.TwoSlopeNorm(vmin=dmin, vcenter=0, vmax=dmax)
    im3 = axes[3].imshow(diff_np, cmap="seismic", norm=norm)
    axes[3].axis("off")
    plt.colorbar(im3, ax=axes[3], pad=0.01, shrink=1.5, aspect=12)

    if not os.path.exists(f"{save_path}/{metric}"):
        os.makedirs(f"{save_path}/{metric}")
    plt.savefig(f"{save_path}/{metric}/{worst_idx}.png")


def save_n_worst_frames_per_metric(
    model,
    n_worst_frames_per_metric: dict[str, tuple[torch.Tensor, ...]],
    save_path: str,
):
    model.eval()
    for metric in n_worst_frames_per_metric.keys():
        worst_cnt = 0
        for frame in n_worst_frames_per_metric[metric]:
            x = frame["input"].unsqueeze(0)
            y = frame["depth"].unsqueeze(0)
            with torch.no_grad():
                y_pred = model(x.to(DEVICE))
                plot_input_and_depth(
                    x, y, y_pred * DEPTH_MAX, save_path, metric, worst_cnt
                )
            worst_cnt += 1


def eval():
    dataset = KittiDataset(  # ../../datasets/kitti_data
        task_paths={"input": "./data/kitti/input", "depth": "./data/kitti/depth/val"},
        task_transform={
            "input": [
                "Crop",
            ],
            "depth": [
                "Crop",
            ],
        },
        camera="image_02",
    )
    eval_dataloader = DataLoader(
        dataset=dataset, batch_size=1, shuffle=False, num_workers=1
    )
    encoder = ResNet18()
    decoder = UnetDecoder()

    model = DepthEncoderDecoder(encoder=encoder, decoder=decoder).to(DEVICE)
    model.load_state_dict(torch.load("../run1/model.pth"))
    model.eval()

    metrics = [MaskedAverageRelativeError(), MaskedRMSE(), MaskedThresholdAccracy()]
    metric_values = {metric.__class__.__name__: [] for metric in metrics}

    for data in tqdm(eval_dataloader, "Eval..."):
        with torch.no_grad():
            pred = model(data["input"].to(DEVICE))
            for metric in metrics:
                metric_values[metric.__class__.__name__].append(
                    metric(pred * DEPTH_MAX, data["depth"].to(DEVICE))
                )

    save_dir = "../train_info/run1"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    worst_n_frames_per_metrics = {metric.__class__.__name__: [] for metric in metrics}
    for metric in metrics:
        _, indices = torch.topk(
            torch.Tensor(metric_values[metric.__class__.__name__]),
            k=3,
            largest=metric.higher,
            sorted=True,
        )
        worst_n_frames_per_metrics[metric.__class__.__name__].extend(
            [dataset[index] for index in indices]
        )
    save_n_worst_frames_per_metric(model, worst_n_frames_per_metrics, save_dir)
    metrics = {
        metric.__class__.__name__: sum(metric_values[metric.__class__.__name__])
        / len(metric_values[metric.__class__.__name__])
        for metric in metrics
    }

    with open(os.path.join(save_dir, "metrics.json"), "w") as file:
        json.dump(metrics, file)

    # TODO:


if __name__ == "__main__":
    eval()
