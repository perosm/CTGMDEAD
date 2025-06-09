import torch
import logging
import eval

from tqdm import tqdm
from utils.shared.aggregators.LossAggregator import LossAggregator
from utils.shared.savers.LossSaver import LossSaver


from utils.shared.utils import (
    prepare_save_directories,
    configure_dataset,
    configure_dataloader,
    configure_model,
    freeze_model,
    configure_optimizer,
    configure_loss,
    configure_logger,
    move_data_to_gpu,
)


def train(args: dict):
    logger = logging.getLogger(__name__)
    save_dir = prepare_save_directories(args, "train")
    logger = configure_logger(save_dir=save_dir, module_name=__name__)
    logging.basicConfig(
        filename=save_dir / "train.log", encoding="utf-8", level=logging.INFO
    )

    device = args["device"]
    dataset = configure_dataset(args["dataset"], mode="train")
    train_dataloader = configure_dataloader(args["train"]["dataloader"], dataset)

    model = configure_model(args["model"], device).to(device)
    losses = configure_loss(args["loss"])
    optimizer = configure_optimizer(model, args["optimizer"])
    epochs = args["epochs"]
    model.train()
    loss_aggregator = LossAggregator(
        task_losses=losses.task_losses, epochs=epochs, num_batches=1, device=device
    )
    loss_saver = LossSaver(loss_aggregator=loss_aggregator, save_dir=save_dir)

    data = next(iter(train_dataloader))
    for epoch in tqdm(range(epochs), "Training..."):
        freeze_model(model, args["model"], epoch)
        data = move_data_to_gpu(data)
        pred = model(data["input"])
        loss, per_batch_task_losses = losses(pred, data)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_aggregator.aggregate_per_batch(per_batch_task_losses)
        logger.log(
            logging.INFO,
            f"epoch: {epoch}; loss: {loss.item()}, per_batch_task_losses: {per_batch_task_losses}",
        )
        if epoch % 50 == 0:
            print(f"Epoch: {epoch}")
            eval.eval(args, model, epoch)
            model.train()

    loss_saver.save_plot()
    torch.save(model.state_dict(), save_dir / "model.pth")

    return
