import os

import torch
import logging
import matplotlib.pyplot as plt
import pprint
from utils.aggregators.LossAggregator import LossAggregator
from utils.savers.LossSaver import LossSaver
from tqdm import tqdm


from utils.kitti.utils import (
    prepare_save_directories,
    configure_dataset,
    configure_dataloader,
    configure_model,
    freeze_model,
    configure_optimizer,
    configure_loss,
    configure_savers,
)


def plot_input_and_depth(image, depth, pred):
    image_np = image.numpy().squeeze(0).transpose(1, 2, 0) / 256.0
    depth_np = depth.numpy().squeeze(0).transpose(1, 2, 0)
    pred_np = pred.numpy().squeeze(0).transpose(1, 2, 0)
    fig, axes = plt.subplots(1, 3, figsize=(24, 16))

    im1 = axes[0].imshow(image_np)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    im2 = axes[1].imshow(depth_np, cmap="viridis")
    axes[1].set_title("Depth Map")
    axes[1].axis("off")
    plt.colorbar(im2, ax=axes[1])

    im3 = axes[2].imshow(pred_np, cmap="viridis")
    axes[2].set_title("Depth Map")
    axes[2].axis("off")
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    plt.show()


def train(args: dict):
    logger = logging.getLogger(__name__)
    save_dir = prepare_save_directories(args, "train")
    logging.basicConfig(
        filename=save_dir / "train.log", encoding="utf-8", level=logging.INFO
    )

    device = args["device"]
    dataset = configure_dataset(args["dataset"])
    train_dataloader = configure_dataloader(args["train"]["dataloader"], dataset)

    model = configure_model(args["model"]).to(device)
    losses = configure_loss(args["loss"])
    optimizer = configure_optimizer(model, args["optimizer"])
    epochs = args["epochs"]
    freeze_model(model, args["model"], True, 0)
    model.train()
    loss_aggregator = LossAggregator(
        task_losses=losses.task_losses, epochs=epochs, num_batches=1, device=device
    )
    loss_saver = LossSaver(
        loss_aggregator=loss_aggregator, save_dir=save_dir
    )  # configure_savers(args["train"]["savers"]) TODO:

    data = next(iter(train_dataloader))
    for epoch in range(epochs):
        freeze_model(model, args["model"], False, epoch)
        # for data in tqdm(train_dataloader, f"Epoch {epoch}"):
        data = {task: data[task].to(device) for task in data.keys()}
        pred = model(data["input"])
        loss, per_batch_task_losses = losses(pred, data)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_aggregator.aggregate_per_batch(per_batch_task_losses)

    loss_saver.save_plot()
    torch.save(model.state_dict(), save_dir / "model.pth")

    return


if __name__ == "__main__":
    train()
