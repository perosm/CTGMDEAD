from tqdm import tqdm
from torch import nn

from utils.shared.utils import (
    configure_dataset,
    configure_dataloader,
    configure_model,
    configure_metrics,
    configure_prediction_postprocessor,
    prepare_save_directories,
    move_data_to_gpu,
)
from utils.shared.visual_inspection import (
    plot_object_detection_predictions_2d,
    plot_projected_height,
    plot_od_3d_output,
)
from utils.shared.aggregators.MetricsAggregator import MetricsAggregator
from utils.shared.savers.MetricSavers import MetricsSaver


def eval(args: dict, model: nn.Module, epoch: int | None):
    device = args["device"]
    save_dir = prepare_save_directories(args, "eval")
    dataset = configure_dataset(args["dataset"])
    eval_dataloader = configure_dataloader(args["eval"]["dataloader"], dataset)

    prediction_postprocessor = configure_prediction_postprocessor(tasks=args["tasks"])
    metrics = configure_metrics(args["metrics"])
    metrics_aggregator = MetricsAggregator(
        task_metrics=metrics.task_metrics, num_batches=1, device=device
    )
    metrics_saver = MetricsSaver(
        metrics_aggregator=metrics_aggregator,
        save_dir=save_dir,
        name=f"metrics{epoch}" if epoch else "metrics",
    )

    model.eval()
    data = next(iter(eval_dataloader))
    # for data in tqdm(eval_dataloader, f"Epoch {epoch}"):
    data = move_data_to_gpu(data)
    pred = model(data["input"])
    # pred = prediction_postprocessor(pred, data)
    images_dir = save_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    # plot_object_detection_predictions_2d(
    #     data["input"],
    #     pred["object_detection_2d"],
    #     data["object_detection_2d"],
    #     images_dir / f"{epoch}.png",
    # )
    plot_od_3d_output(pred=pred, data=data)
    model.train()
    per_batch_task_metrics = metrics(pred, data)
    metrics_aggregator.aggregate_per_batch(per_batch_task_metrics)

    metrics_saver.save()
