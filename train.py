import os

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset.kitti.KittiDataset import KittiDataset
from model.resnet import ResNet18
from model.decoder import UnetDecoder
from model.encoder_decoder import DepthEncoderDecoder
from utils.losses import MaskedMAE, GradLoss

from utils.kitti.utils import freeze_params, print_model_size

DEPTH_MAX = 90.0


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


def train():
    device = "cuda"
    dataset = KittiDataset(  # ../../datasets/kitti_data
        task_paths={"input": "./data/kitti/input", "depth": "./data/kitti/depth/train"},
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
    train_dataloader = DataLoader(
        dataset=dataset, batch_size=2, shuffle=True, num_workers=1
    )
    encoder = ResNet18()
    decoder = UnetDecoder()

    model = DepthEncoderDecoder(encoder=encoder, decoder=decoder).to(device)
    print_model_size(model)
    freeze_params(model.encoder, True)
    masked_mae = MaskedMAE()
    # grad_loss = GradLoss(device)
    optimizer = Adam(model.parameters(), lr=1e-4)

    data = next(iter(train_dataloader))

    epochs = 1
    model.train()
    losses = {}
    cnt_print = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for data in tqdm(train_dataloader, f"Epoch {epoch}"):
            pred = model(data["input"].to(device))
            loss = 1 * masked_mae(
                torch.clamp(pred * DEPTH_MAX, 0, DEPTH_MAX), data["depth"].to(device)
            )  # + 1 * grad_loss(pred * DEPTH_MAX, gt)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if epoch == 10:
                freeze_params(model.encoder, False)

            if cnt_print % 1000:
                print(f"loss={loss}")
            cnt_print += 1

            epoch_loss += loss

        losses[epoch] = str(epoch_loss / len(train_dataloader))
        print(f"Epoch loss={losses[epoch]}")

    save_dir = "../train_info/run1"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Spremanje modela i gubitaka
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))
    with open(os.path.join(save_dir, "losses.txt"), "w") as f:
        f.writelines(losses)


if __name__ == "__main__":
    train()
