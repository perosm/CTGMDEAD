import torch
import logging
import eval

from tqdm import tqdm
from utils.shared.aggregators.LossAggregator import LossAggregator
from utils.shared.savers.LossSaver import LossSaver
from utils.shared.early_stopping import EarlyStopping
from utils.shared.model_saver import ModelSaver

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
    configure_metrics,
    remove_dummy_ground_truth,
)


def train(args: dict):
    # train specific stuff
    logger = logging.getLogger(__name__)
    train_save_dir = prepare_save_directories(args, "train")
    val_save_dir = prepare_save_directories(args, "eval")
    logger = configure_logger(save_dir=train_save_dir, module_name=__name__)
    logging.basicConfig(
        filename=train_save_dir / "train.log", encoding="utf-8", level=logging.INFO
    )

    device = args["device"]
    dataset = configure_dataset(args["dataset"], mode="train")
    train_dataloader = configure_dataloader(args["train"]["dataloader"], dataset)

    model = configure_model(args["model"], device).to(device)
    losses = configure_loss(args["loss"])
    optimizer = configure_optimizer(model, args["optimizer"])
    epochs = args["epochs"]
    loss_aggregator = LossAggregator(
        task_losses=losses.task_losses,
        epochs=epochs,
        num_batches_total=len(train_dataloader),
        device=device,
    )
    loss_saver = LossSaver(
        loss_aggregator=loss_aggregator, save_dir=train_save_dir, device=device
    )
    early_stopping = EarlyStopping(**args["early_stopping"])
    # val specific stuff
    model_saver = ModelSaver(
        save_dir=val_save_dir.parent,
        task_metrics=configure_metrics(args["metrics"]).task_metrics,
    )
    val_loss_aggregator = LossAggregator(
        task_losses=losses.task_losses,
        epochs=epochs,
        num_batches_total=1,  # will be set to len(val_dataloader)
        device="cpu",
    )
    val_loss_saver = LossSaver(
        loss_aggregator=val_loss_aggregator, save_dir=val_save_dir
    )
    for epoch in tqdm(range(epochs), f"Training..."):
        model.train()
        freeze_model(model, args["model"], epoch)
        for data in tqdm(train_dataloader, f"Epoch: {epoch}"):
            data = move_data_to_gpu(data)
            data = remove_dummy_ground_truth(data)
            pred = model(data["input"])
            loss, per_batch_task_losses = losses(pred, data)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_aggregator.aggregate_per_batch(per_batch_task_losses)

        logger.log(
            logging.INFO,
            f"epoch {epoch}: loss={loss_aggregator.total_loss_per_epochs[epoch].item()}",
        )
        eval.eval(args, model, epoch, early_stopping, val_loss_aggregator, model_saver)
        if early_stopping.early_stop:
            logger.log(logging.INFO, f"Early stopping at epoch {epoch}!")
            break

    loss_saver.save()
    loss_saver.save_plot()
    val_loss_saver.save()
    val_loss_saver.save_plot()
