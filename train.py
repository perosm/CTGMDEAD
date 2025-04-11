import torch
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.visual_inspection import plot_task_gt
from utils.aggregators.LossAggregator import LossAggregator
from utils.savers.LossSaver import LossSaver


from utils.kitti.utils import (
    prepare_save_directories,
    configure_dataset,
    configure_dataloader,
    configure_model,
    freeze_model,
    configure_optimizer,
    configure_loss,
)


def train(args: dict):
    logger = logging.getLogger(__name__)
    save_dir = prepare_save_directories(args, "train")
    logging.basicConfig(
        filename=save_dir / "train.log", encoding="utf-8", level=logging.INFO
    )

    device = args["device"]
    dataset = configure_dataset(args["dataset"])
    train_dataloader = configure_dataloader(args["train"]["dataloader"], dataset)

    # model = configure_model(args["model"]).to(device)
    losses = configure_loss(args["loss"])
    # optimizer = configure_optimizer(model, args["optimizer"])
    epochs = args["epochs"]
    # freeze_model(model, args["model"], True, 0)
    # model.train()
    loss_aggregator = LossAggregator(
        task_losses=losses.task_losses, epochs=epochs, num_batches=1, device=device
    )
    loss_saver = LossSaver(loss_aggregator=loss_aggregator, save_dir=save_dir)

    # data = next(iter(train_dataloader))
    for epoch in range(epochs):
        # freeze_model(model, args["model"], False, epoch)
        for data in tqdm(train_dataloader, f"Epoch {epoch}"):
            data = {task: data[task].to(device) for task in data.keys()}
            plot_task_gt(data)
        # pred = model(data["input"])
        # loss, per_batch_task_losses = losses(pred, data)
        # loss.backward()
        # optimizer.step()
        # optimizer.zero_grad()
        # loss_aggregator.aggregate_per_batch(per_batch_task_losses)
    loss_saver.save_plot()
    # torch.save(model.state_dict(), save_dir / "model.pth")

    return


if __name__ == "__main__":
    train()
