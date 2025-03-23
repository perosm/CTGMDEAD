import os
import re
import pathlib

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.optim import Optimizer, Adam
import torchvision.transforms.functional as F


from dataset.kitti.KittiDataset import KittiDataset
from model.resnet import ResNet18, ResNet
from model.decoder import UnetDecoder
from model.multi_task_network import MultiTaskNetwork
from utils.losses import GradLoss, MaskedMAE, MultiTaskLoss


def prepare_save_directories(args: dict, subfolder_name="train") -> None:
    save_dir = pathlib.Path(
        os.path.join(args["save_path"], args["save_path"], subfolder_name)
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


############################## MODEL UTILS ##############################
def configure_model(model_configs: dict) -> nn.Module:
    encoder = _configure_encoder(model_configs["encoder"])
    decoders = _configure_decoder(model_configs["decoder"])
    model = MultiTaskNetwork(encoder, decoders)
    print_model_size(model)

    return model


def _configure_encoder(encoder_configs: dict) -> nn.Module:
    encoder_dict = {f"{ResNet.__name__}18": ResNet18(encoder_configs["pretrained"])}

    return encoder_dict[encoder_configs["name"]]


def _configure_decoder(decoder_configs: dict) -> nn.Module:
    decoder_task = {}
    decoder_dict = {UnetDecoder.__name__: UnetDecoder}
    for task in decoder_configs.keys():
        decoder_task_configs = decoder_configs[task]
        decoder_task[task] = decoder_dict[decoder_task_configs["name"]](
            decoder_task_configs["in_channels"],
            decoder_task_configs["channel_scale_factors"],
            decoder_task_configs["out_channels"],
        )

    return decoder_task


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
        if model_configs["decoder"][task][command] == epoch:
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
    loss_dict = {MaskedMAE.__name__: MaskedMAE(), GradLoss.__name__: GradLoss()}
    task_losses = {task: [] for task in loss_configs.keys()}
    for task in loss_configs.keys():
        for loss_name in loss_configs[task]:
            task_losses[task].append(loss_dict[loss_name])

    return MultiTaskLoss(task_losses)


def configure_optimizer(model: nn.Module, optimizer_configs: dict) -> Optimizer:
    optimizer_dict = {Adam.__name__: Adam}

    return optimizer_dict[optimizer_configs["name"]](
        model.parameters(), float(optimizer_configs["lr"])
    )


############################## TEST UTILS ##############################
def configure_metrics(metric_configs):
    pass
