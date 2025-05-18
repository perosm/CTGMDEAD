import os
import re
import pathlib
import logging

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.optim import Optimizer, Adam
from utils.shared.savers.Saver import Saver
from utils.shared.savers.LossSaver import LossSaver
from utils.shared.aggregators.Aggregator import Aggregator
from utils.shared.metrics import MultiTaskMetrics

from dataset.kitti.KittiDataset import KittiDataset
from model.resnet import ResNet18, ResNet
from model.depth_estimation.depth_decoder import UnetDepthDecoder
from model.road_detection.road_detection_decoder import UnetRoadDetectionDecoder
from model.object_detection.fpn_faster_rcnn import FPNFasterRCNN
from model.multi_task_network import MultiTaskNetwork
from utils.shared.enums import TaskEnum
from utils.shared.losses import (
    GradLoss,
    MaskedMAE,
    MultiTaskLoss,
    BinaryCrossEntropyLoss,
)
from utils.object_detection.losses import (
    RPNClassificationAndRegressionLoss,
    RCNNCrossEntropyAndRegressionLoss,
)

from utils.shared.prediction_postprocessor import PredictionPostprocessor
from utils.object_detection.prediction_postprocessor import (
    PredictionPostprocessor as ObjectDetection2DPredictionPostprocessor,
)
from utils.depth.prediction_postprocessor import (
    PredictionPostprocessor as DepthPredictionPostProcessor,
)
from utils.road_detection.prediction_postprocessor import (
    PredictionPostprocessor as RoadPredictionPostprocessor,
)
from utils.depth.metrics import (
    MaskedAverageRelativeError,
    MaskedMeanAbsoluteError,
    MaskedRMSE,
    MaskedThresholdAccracy,
)

from utils.road_detection.metrics import (
    IoU,
    Precision,
    Recall,
    FalsePositiveRate,
    TrueNegativeRate,
)
from utils.object_detection.metrics import mAP


def prepare_save_directories(args: dict, subfolder_name="train") -> None:
    save_dir = pathlib.Path(
        os.path.join(args["save_path"], args["name"], subfolder_name)
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    return save_dir


############################## CONFIG UTILS ##############################
def configure_dataset(dataset_configs: dict[str, str | list]) -> Dataset:
    dataset_dict = {
        KittiDataset.__name__: KittiDataset(
            dataset_configs["task_paths"], dataset_configs["task_transform"]
        )
    }
    return dataset_dict[dataset_configs["dataset_name"]]


def configure_dataloader(
    dataloader_configs: dict[str, str | bool], dataset: Dataset
) -> DataLoader:
    return DataLoader(
        dataset=dataset,
        batch_size=dataloader_configs["batch_size"],
        shuffle=dataloader_configs["shuffle"],
        num_workers=dataloader_configs["num_workers"],
    )


def configure_savers(
    savers_configs: dict[str, str], aggregators: dict[str, Aggregator]
) -> list[Saver]:
    savers = []
    savers_dict = {LossSaver.__class__.__name__: LossSaver}
    for aggregator_name, saver_name in savers_configs.items():
        savers.append(savers_dict[saver_name(aggregators[aggregator_name])])

    return savers


def configure_logger(save_dir: pathlib.Path, module_name: str) -> logging.Logger:
    # https://docs.python.org/3/howto/logging-cookbook.html
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(save_dir / f"{module_name}.log")
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    terminal_handler = logging.StreamHandler()
    terminal_handler.setLevel(logging.INFO)
    logger.addHandler(terminal_handler)

    return logger


############################## MODEL UTILS ##############################
def configure_model(model_configs: dict, device: torch.device) -> nn.Module:
    encoder = _configure_encoder(model_configs["encoder"]).to(device)
    depth_decoder = _configure_decoder(model_configs["depth_decoder"]).to(device)
    road_detection_decoder = _configure_decoder(
        model_configs.get("road_detection_decoder", None)
    )
    if road_detection_decoder:
        road_detection_decoder = road_detection_decoder.to(device)

    necks_and_heads = _configure_necks_and_heads(
        model_configs.get("necks_and_heads", None), device
    )
    model = MultiTaskNetwork(
        encoder=encoder,
        depth_decoder=depth_decoder,
        road_detection_decoder=road_detection_decoder,
        heads_and_necks=necks_and_heads,
    )
    print_model_size(model)

    return model


def _configure_encoder(encoder_configs: dict) -> nn.Module:
    encoder_dict = {f"{ResNet.__name__}18": ResNet18(encoder_configs["pretrained"])}

    return encoder_dict[encoder_configs["name"]]


def _configure_decoder(decoder_configs: dict) -> nn.Module:
    decoder_dict = {
        UnetDepthDecoder.__name__: UnetDepthDecoder,
        UnetRoadDetectionDecoder.__name__: UnetRoadDetectionDecoder,
    }

    if not decoder_configs:
        return None

    return decoder_dict[decoder_configs["name"]](
        decoder_configs["in_channels"],
        decoder_configs["channel_scale_factors"],
        decoder_configs["out_channels"],
    )


def _configure_necks_and_heads(
    necks_and_heads_configs: dict, device: torch.device
) -> nn.Module:
    necks_and_heads = {}
    necks_and_heads_dict = {FPNFasterRCNN.__name__: FPNFasterRCNN}

    if not necks_and_heads_configs:
        return None

    for task in necks_and_heads_configs.keys():
        necks_and_heads_info = necks_and_heads_configs[task]
        name = necks_and_heads_info.pop("name")
        necks_and_heads[task] = necks_and_heads_dict[name](
            necks_and_heads_configs[task]
        ).to(device)

    return necks_and_heads


def print_model_size(model: nn.Module) -> None:
    """
    Returns size of model in megabytes.

    Args:
        - model (nn.Module): pytorch model which size we wish to know.

    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_in_mb = (param_size + buffer_size) / 1024**2

    print(f"model size: {size_in_mb:.3f}MB")


def freeze_model(
    model: MultiTaskNetwork, model_configs: dict, freeze: bool, epoch: int = 0
) -> None:
    command = "freeze_epoch" if freeze else "unfreeze_epoch"
    if model_configs["encoder"][command] == epoch:
        freeze_params(model.encoder, freeze)

    if model_configs["depth_decoder"][command] == epoch:
        freeze_params(model.depth_decoder, freeze)

    road_detection_decoder_configs = model_configs.get("road_detection_decoder", None)
    if road_detection_decoder_configs:
        if road_detection_decoder_configs[command] == epoch:
            freeze_params(model.road_detection_decoder, freeze)


def freeze_params(
    model: nn.Module, freeze=True, layers: dict[str, list[str]] = {"*": ["*"]}
) -> None:
    """
    Freezes all layers defined in the layers dict.
    By default we all layers are frozen.

    Args:
     - model (nn.Module): model whose layers are to be freezed
     - freeze (bool): freeze flag. If true freezes given layers. If false unfreezes given layers.
     - layers (Dict[str, List[str]]): dictionary whose keys represent top level building blocks
                                      and whose values represent lower level components such as
                                      convolutions, batchnorm etc...

    Returns:
     - model (nn.Module): model with frozen/unfrozen layers.
    """
    pattern = None
    for layer_name, sublayer_names in layers.items():
        layer_pattern = re.sub(r"\*", r".*", layer_name)
        for sublayer_name in sublayer_names:
            if sublayer_name == "*":
                pattern = layer_pattern
            else:
                sublayer_pattern = re.sub(r"\*", ".*", sublayer_name)
                pattern = f"{layer_pattern}.*{sublayer_pattern}"
            for name, param in model.named_parameters():
                if re.search(pattern, name, re.DOTALL):
                    print(f"{name}" + (" frozen!" if freeze else " unfrozen!"))
                    param.requires_grad = False if freeze else True


############################## TRAIN UTILS ##############################
def configure_loss(loss_configs: dict) -> MultiTaskLoss:
    loss_dict = {
        MaskedMAE.__name__: MaskedMAE,
        GradLoss.__name__: GradLoss,
        BinaryCrossEntropyLoss.__name__: BinaryCrossEntropyLoss,
        RPNClassificationAndRegressionLoss.__name__: RPNClassificationAndRegressionLoss,
        RCNNCrossEntropyAndRegressionLoss.__name__: RCNNCrossEntropyAndRegressionLoss,
    }
    task_losses = {task: [] for task in loss_configs.keys()}
    for task in loss_configs.keys():
        for loss_name in loss_configs[task]:
            task_losses[task].append(loss_dict[loss_name]())

    return MultiTaskLoss(task_losses)


def configure_optimizer(model: nn.Module, optimizer_configs: dict) -> Optimizer:
    optimizer_dict = {Adam.__name__: Adam}

    return optimizer_dict[optimizer_configs["name"]](
        model.parameters(), float(optimizer_configs["lr"])
    )


def configure_prediction_postprocessor(tasks: list[str]):
    per_task_postprocessing_funcs = {}
    postprocess_functions = {
        TaskEnum.depth: DepthPredictionPostProcessor,
        TaskEnum.road_detection: RoadPredictionPostprocessor,
        TaskEnum.object_detection_2d: ObjectDetection2DPredictionPostprocessor,
    }
    for task in tasks:
        per_task_postprocessing_funcs[task] = postprocess_functions[task]()

    return PredictionPostprocessor(
        per_task_postprocessing_funcs=per_task_postprocessing_funcs
    )


############################## TEST UTILS ##############################
def configure_metrics(metric_configs):
    metrics_dict = {
        # depth metrics
        MaskedAverageRelativeError.__name__: MaskedAverageRelativeError,
        MaskedMAE.__name__: MaskedMAE,
        MaskedMeanAbsoluteError.__name__: MaskedMeanAbsoluteError,
        MaskedRMSE.__name__: MaskedRMSE,
        MaskedThresholdAccracy.__name__: MaskedThresholdAccracy,
        # road detection metrics
        IoU.__name__: IoU,
        Precision.__name__: Precision,
        Recall.__name__: Recall,
        TrueNegativeRate.__name__: TrueNegativeRate,
        FalsePositiveRate.__name__: FalsePositiveRate,
        # object detection metrics
        mAP.__name__: mAP,
    }
    task_metrics = {task: [] for task in metric_configs.keys()}
    for task in metric_configs.keys():
        for metric_name in metric_configs[task]:
            task_metrics[task].append(metrics_dict[metric_name]())

    return MultiTaskMetrics(task_metrics)
