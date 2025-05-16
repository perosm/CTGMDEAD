from tqdm import tqdm
from torch import nn

from utils.shared.utils import (
    configure_dataset,
    configure_dataloader,
    configure_model,
    configure_metrics,
    configure_prediction_postprocessor,
    prepare_save_directories,
)

from utils.shared.aggregators.MetricsAggregator import MetricsAggregator
from utils.shared.savers.MetricSavers import MetricsSaver


def eval(args: dict, model: nn.Module, epoch: int | None):
    device = args["device"]
    save_dir = prepare_save_directories(args, "eval")
    dataset = configure_dataset(args["dataset"])
    eval_dataloader = configure_dataloader(args["eval"]["dataloader"], dataset)

    # prediction_postprocessor = configure_prediction_postprocessor(tasks=args["tasks"])
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
    data = {task: data[task].to(device) for task in data.keys()}
    pred = model(data["input"])
    # pred = prediction_postprocessor(pred)
    per_batch_task_metrics = metrics(pred, data)
    metrics_aggregator.aggregate_per_batch(per_batch_task_metrics)

    metrics_saver.save()
