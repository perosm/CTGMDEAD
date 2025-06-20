import argparse
import pathlib

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
    configure_model,
    load_yaml_file,
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
    loss_aggregator.num_batches_total = len(eval_dataloader)

    metrics = configure_metrics(args["metrics"])
    metrics_aggregator = MetricsAggregator(
        task_metrics=metrics.task_metrics,
        num_batches_total=len(eval_dataloader),
        device=device,
    )
    metrics_saver = MetricsSaver(
        metrics_aggregator=metrics_aggregator,
        save_dir=save_dir,
        name=f"metrics_{epoch}" if epoch else "metrics",
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

        if early_stopping:
            early_stopping(loss_aggregator.total_loss_per_epochs[epoch].item())

    metrics_saver.save()
    if model_saver:
        model_saver(model, metrics_aggregator.task_metrics_per_epochs)
    visualizers.plot_visualizations(pred=pred, gt=data, image=data["input"])

    return


def main():
    args = _parse_args()

    # Load .yaml config file
    config_file_path = args.config_file_path
    config_file = load_yaml_file(config_file_path)
    device = config_file["device"]

    # Using save_path and .yaml file name fetch model weights
    save_dir = config_file["save_path"]
    config_filename = config_file_path.stem
    config_file.update({"name": config_filename})
    model_weights_path = pathlib.Path(save_dir, config_filename, "best_model.pth")
    config_file["model"].update({"weights_file_path": model_weights_path})
    model = configure_model(config_file["model"], device).to(device)

    # Initialize loss aggregator
    losses = configure_loss(config_file["loss"])
    loss_aggregator = LossAggregator(
        task_losses=losses.task_losses,
        epochs=1,
        num_batches_total=1,
    )
    eval(
        args=config_file,
        model=model,
        epoch="best",
        early_stopping=None,
        loss_aggregator=loss_aggregator,
        model_saver=None,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cfp",
        "--config-file-path",
        type=pathlib.Path,
        default=pathlib.Path(
            "./configs/configs_info/00_ResNet18_kitti_depth_estimation.yaml"
        ),
        help="Path to config file.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
