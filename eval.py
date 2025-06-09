import torch
from torch import nn
from tqdm import tqdm

from utils.shared.utils import (
    configure_dataset,
    configure_dataloader,
    configure_model,
    configure_metrics,
    configure_eval_prediction_postprocessor,
    prepare_save_directories,
    move_data_to_gpu,
    configure_visualizers,
)
from utils.shared.aggregators.MetricsAggregator import MetricsAggregator
from utils.shared.savers.MetricSavers import MetricsSaver


def eval(args: dict, model: nn.Module, epoch: int | None):
    device = args["device"]
    save_dir = prepare_save_directories(args, "eval")
    dataset = configure_dataset(args["dataset"], mode="val")
    eval_dataloader = configure_dataloader(args["eval"]["dataloader"], dataset)

    prediction_postprocessor = configure_eval_prediction_postprocessor(
        task_postprocess_infos=args["prediction_postprocess"]
    )
    # metrics = configure_metrics(args["metrics"])
    # metrics_aggregator = MetricsAggregator(
    #     task_metrics=metrics.task_metrics, num_batches=1, device=device
    # )
    # metrics_saver = MetricsSaver(
    #     metrics_aggregator=metrics_aggregator,
    #     save_dir=save_dir,
    #     name=f"metrics{epoch}" if epoch else "metrics",
    # )
    visualizers = configure_visualizers(args, save_dir, epoch)

    model.eval()
    with torch.no_grad():
        data = next(iter(eval_dataloader))
        # for data in tqdm(eval_dataloader, f"Epoch {epoch}"):
        data = move_data_to_gpu(data)
        pred = model(data["input"])
        pred = prediction_postprocessor(pred, data["projection_matrix"])
        visualizers.plot_visualizations(pred=pred, gt=data, image=data["input"])

        # per_batch_task_metrics = metrics(pred, data)
        # metrics_aggregator.aggregate_per_batch(per_batch_task_metrics)

        # metrics_saver.save()

    return
