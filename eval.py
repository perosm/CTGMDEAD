import torch
from torch import nn
from tqdm import tqdm

from utils.shared.utils import (
    configure_dataset,
    configure_dataloader,
    configure_loss,
    configure_metrics,
    configure_eval_prediction_postprocessor,
    prepare_save_directories,
    move_data_to_gpu,
    configure_visualizers,
)
from utils.shared.aggregators.LossAggregator import LossAggregator
from utils.shared.savers.LossSaver import LossSaver
from utils.shared.aggregators.MetricsAggregator import MetricsAggregator
from utils.shared.savers.MetricSavers import MetricsSaver
from utils.shared.early_stopping import EarlyStopping
from utils.shared.model_saver import ModelSaver
from utils.shared.utils import remove_dummy_ground_truth


def eval(
    args: dict,
    model: nn.Module,
    epoch: int | None,
    early_stopping: EarlyStopping,
    loss_aggregator: LossAggregator,
    model_saver: ModelSaver,
):
    device = args["device"]
    save_dir = prepare_save_directories(args, "eval")
    dataset = configure_dataset(args["dataset"], mode="val")
    eval_dataloader = configure_dataloader(args["eval"]["dataloader"], dataset)

    prediction_postprocessor = configure_eval_prediction_postprocessor(
        task_postprocess_infos=args["prediction_postprocess"]
    )

    losses = configure_loss(args["loss"])
    loss_aggregator.num_batches = len(eval_dataloader)

    metrics = configure_metrics(args["metrics"])
    metrics_aggregator = MetricsAggregator(
        task_metrics=metrics.task_metrics,
        num_batches_total=len(eval_dataloader),
        device=device,
    )
    metrics_saver = MetricsSaver(
        metrics_aggregator=metrics_aggregator,
        save_dir=save_dir,
        name=f"metrics{epoch}" if epoch else "metrics",
    )
    visualizers = configure_visualizers(args, save_dir, epoch)

    with torch.no_grad():
        data = next(iter(eval_dataloader))
        for data in tqdm(eval_dataloader, f"Validating..."):
            data = move_data_to_gpu(data)
            data = remove_dummy_ground_truth(data)

            model.train()
            loss, per_batch_task_losses = losses(model(data["input"]), data)
            loss_aggregator.aggregate_per_batch(per_batch_task_losses)

            model.eval()
            pred = prediction_postprocessor(
                model(data["input"]), data["projection_matrix"]
            )

            per_batch_task_metrics = metrics(pred, data)
            metrics_aggregator.aggregate_per_batch(per_batch_task_metrics)
        early_stopping(loss_aggregator.total_loss_per_epochs[epoch].item())

    metrics_saver.save()
    model_saver(model, metrics_aggregator.task_metrics_per_epochs)
    visualizers.plot_visualizations(pred=pred, gt=data, image=data["input"])

    return
