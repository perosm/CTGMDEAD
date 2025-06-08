import os
import re
import pathlib
import logging
from typing import Any
from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.optim import Optimizer, Adam, SGD
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
from utils.object_detection_3d.losses import (
    UncertaintyAwareRegressionLoss,
    L1SizeLoss,
    L1YawLoss,
    L1KeypointsLoss,
)

from utils.shared.prediction_postprocessor import PredictionPostprocessor
from utils.depth.prediction_postprocessor import (
    PredictionPostprocessor as DepthPredictionPostProcessor,
)
from utils.road_detection.prediction_postprocessor import (
    PredictionPostprocessor as RoadPredictionPostprocessor,
)
from utils.object_detection.prediction_postprocessor import (
    PredictionPostprocessor as ObjectDetectionDPredictionPostprocessor,
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
from utils.shared.visualizer import Visualizer
from utils.object_detection.visualizer import Visualizer as ObjectDetectionVisualizer
from utils.depth.visualizer import Visualizer as DepthVisualizer
from utils.road_detection.visualizer import Visualizer as RoadVisualizer


def prepare_save_directories(args: dict, subfolder_name="train") -> None:
    save_dir = pathlib.Path(
        os.path.join(args["save_path"], args["name"], subfolder_name)
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    return save_dir


from utils.shared.dict_utils import list_of_dict_to_dict


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
    _load_model_weights()
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
    necks_and_heads_dict = {FPNFasterRCNN.__name__: FPNFasterRCNN}

    if not necks_and_heads_configs:
        return None

    return FPNFasterRCNN(necks_and_heads_configs)


def _load_model_weights():  # TODO:
    pass


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


def freeze_model(model: MultiTaskNetwork, model_configs: dict, epoch: int = 0) -> None:
    submodules_dict = {
        "encoder": model.encoder,
        "depth_decoder": model.depth_decoder,
        "road_detection_decoder": getattr(model, "road_detection_decoder", None),
    }

    for submodule_name, submodule in submodules_dict.items():
        submodule_config = list_of_dict_to_dict(
            model_configs.get(submodule_name, {}), {}, 1
        )
        if submodule:
            if submodule_config.get("freeze_epoch") == epoch:
                freeze_params(submodule, freeze=True)
            if submodule_config.get("unfreeze_epoch") == epoch:
                freeze_params(submodule, freeze=False)

    heads_and_necks_config = model_configs.get("necks_and_heads", {})
    heads_and_necks = getattr(model, "heads_and_necks", None)

    if heads_and_necks and heads_and_necks_config:
        for submodule_name in [
            "rpn",
            "roi",
            "output_heads",
            "distance_head",
            "attribute_head",
        ]:
            submodule = getattr(heads_and_necks, submodule_name, None)
            submodule_config = list_of_dict_to_dict(
                heads_and_necks_config.get(submodule_name, {})
            )
            if submodule:
                if submodule_config.get("freeze_epoch") == epoch:
                    freeze_params(submodule, freeze=True)
                if submodule_config.get("unfreeze_epoch") == epoch:
                    freeze_params(submodule, freeze=False)


def freeze_params(
    model: nn.Module, freeze=True, layers: dict[str, list[str]] = {"*": ["*"]}
) -> None:
    """
    Freezes all layers defined in the layers dict.
    By default we all layers are frozen.

    Args:
     - model: Module whose layers are to be freezed or unfreezed.
     - freeze: If True freezes given layers. If False unfreezes given layers.
     - layers: Dictionary whose keys represent top level building blocks
               and whose values represent lower level components such as
               convolutions, batchnorm etc...
    """
    print(f"Module: {model._get_name()}")
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
                    print(f"\t{name}" + (" frozen!" if freeze else " unfrozen!"))
                    param.requires_grad = False if freeze else True


############################## TRAIN UTILS ##############################
def configure_loss(loss_configs: dict) -> MultiTaskLoss:
    loss_dict = {
        # depth losses
        MaskedMAE.__name__: MaskedMAE,
        GradLoss.__name__: GradLoss,
        # road detection losses
        BinaryCrossEntropyLoss.__name__: BinaryCrossEntropyLoss,
        # object detection 2d losses
        RPNClassificationAndRegressionLoss.__name__: RPNClassificationAndRegressionLoss,
        RCNNCrossEntropyAndRegressionLoss.__name__: RCNNCrossEntropyAndRegressionLoss,
        # object detection 3d losses
        UncertaintyAwareRegressionLoss.__name__: UncertaintyAwareRegressionLoss,
        L1SizeLoss.__name__: L1SizeLoss,
        L1YawLoss.__name__: L1YawLoss,
        L1KeypointsLoss.__name__: L1KeypointsLoss,
    }
    task_losses = {task: [] for task in loss_configs.keys()}
    for task in loss_configs.keys():
        for loss_name in loss_configs[task]:
            task_losses[task].append(loss_dict[loss_name]())

    return MultiTaskLoss(task_losses)


def configure_optimizer(model: nn.Module, optimizer_configs: dict) -> Optimizer:
    optimizer_dict = {Adam.__name__: Adam, SGD.__name__: SGD}

    return optimizer_dict[optimizer_configs["name"]](
        model.parameters(), float(optimizer_configs["lr"])
    )


def configure_eval_prediction_postprocessor(
    task_postprocess_infos: list[dict[str, dict[str, Any] | None]],
) -> PredictionPostprocessor:
    per_task_postprocessing_funcs = {}
    postprocess_functions = {
        TaskEnum.depth: DepthPredictionPostProcessor,
        TaskEnum.road_detection: RoadPredictionPostprocessor,
        TaskEnum.object_detection: ObjectDetectionDPredictionPostprocessor,
    }
    merged_postprocess_info = defaultdict(dict)
    for task_postprocess_info in task_postprocess_infos:
        for task, postprocess_info in task_postprocess_info.items():
            if postprocess_info:
                postprocess_info_dict = list_of_dict_to_dict(postprocess_info, {}, 1)
                merged_postprocess_info[task].update(postprocess_info_dict)
            else:
                # To ensure the key is inside the defaultdict
                merged_postprocess_info[task]

    for task, postprocess_info in merged_postprocess_info.items():
        per_task_postprocessing_funcs[task] = (
            postprocess_functions[task](**postprocess_info)
            if postprocess_info
            else postprocess_functions[task]()
        )

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
        # object detection 2D metrics
        mAP.__name__: mAP,
        # object detection 3D metrics
    }
    task_metrics = {task: [] for task in metric_configs.keys()}
    for task in metric_configs.keys():
        for metric_name in metric_configs[task]:
            task_metrics[task].append(metrics_dict[metric_name]())

    return MultiTaskMetrics(task_metrics)


def configure_visualizers(
    configs: dict, save_dir: pathlib.Path, epoch: int
) -> Visualizer:
    visualizer_configs = configs["dataset"]["task_paths"]
    save_path = save_dir / "images"

    visualizers_dict = {
        ObjectDetectionVisualizer.task: ObjectDetectionVisualizer,
        DepthVisualizer.task: DepthVisualizer,
        RoadVisualizer.task: RoadVisualizer,
    }

    return Visualizer(
        {
            task: visualizers_dict[task]()
            for task in visualizer_configs.keys()
            if task in visualizers_dict
        },
        save_dir=save_path,
        epoch=epoch,
    )


############################## SHARED UTILS ##############################
def move_data_to_gpu(data: dict[str, torch.Tensor | dict[str, torch.Tensor]]):
    for task, value in data.items():
        if isinstance(value, torch.Tensor):
            data[task] = value.to(device="cuda")
        elif isinstance(value, dict):
            for subkey, subvalue in value.items():
                if isinstance(subvalue, torch.Tensor):
                    data[task][subkey] = subvalue.to("cuda")

    return data
