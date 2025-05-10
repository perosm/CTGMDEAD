import os
import re
import pathlib
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.optim import Optimizer, Adam
import torchvision.transforms.functional as F
from utils.shared.savers.Saver import Saver
from utils.shared.savers.LossSaver import LossSaver
from utils.shared.aggregators.Aggregator import Aggregator
from utils.shared.aggregators.LossAggregator import LossAggregator

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
    RPNClassificationRegressionLoss,
    RCNNCrossEntropyLoss,
)


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


############################## MODEL UTILS ##############################
def configure_model(model_configs: dict, device: torch.device) -> nn.Module:
    encoder = _configure_encoder(model_configs["encoder"]).to(device)
    decoder = _configure_decoder(model_configs["decoder"]).to(device)
    necks_and_heads = _configure_necks_and_heads(
        model_configs["necks_and_heads"], device
    )
    model = MultiTaskNetwork(
        encoder=encoder, decoder=decoder, heads_and_necks=necks_and_heads
    )
    print_model_size(model)

    return model


def _configure_encoder(encoder_configs: dict) -> nn.Module:
    encoder_dict = {f"{ResNet.__name__}18": ResNet18(encoder_configs["pretrained"])}

    return encoder_dict[encoder_configs["name"]]


def _configure_decoder(decoder_configs: dict) -> nn.Module:
    decoder_dict = {UnetDepthDecoder.__name__: UnetDepthDecoder}

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

    for task in model_configs["decoder"].keys():
        if model_configs["decoder"][command] == epoch:
            freeze_params(model.decoders[task], freeze)


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
        RPNClassificationRegressionLoss.__name__: RPNClassificationRegressionLoss,
        RCNNCrossEntropyLoss.__name__: RCNNCrossEntropyLoss,
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


def prediction_postprocessing(predictions: dict[str, Any]):
    postprocessed_predictions = {}
    postprocess_functions = {
        TaskEnum.depth: True,
    }
    for task, values in predictions.items():
        postprocessed_predictions[task] = postprocess_functions[task](values)

    return postprocessed_predictions


############################## TEST UTILS ##############################
def configure_metrics(metric_configs):
    pass
