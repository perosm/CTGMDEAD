import os
import re
import pathlib
import logging
import yaml
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
from dataset.nuscenes.NuScenesDataset import NuScenesNuImagesDataset
from model.resnet import ResNet18, ResNet
from model.input_reconstruction.input_reconstruction_decoder import (
    UnetInputReconstructionDecoder,
)
from model.depth_estimation.depth_decoder import UnetDepthDecoder
from model.road_detection.road_detection_decoder import UnetRoadDetectionDecoder
from model.object_detection.fpn_faster_rcnn import FPNFasterRCNN
from model.multi_task_network import MultiTaskNetwork
from utils.shared.enums import TaskEnum
from utils.shared.losses import MultiTaskLoss
from utils.input_reconstruction.losses import MSE
from utils.depth.losses import GradLoss, MaskedMAE
from utils.road_detection.losses import BinaryCrossEntropyLoss
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
from utils.input_reconstruction.prediction_postprocessor import (
    PredictionPostprocessor as InputPredictionPostProcessor,
)
from utils.depth.prediction_postprocessor import (
    PredictionPostprocessor as DepthPredictionPostProcessor,
)
from utils.road_detection.prediction_postprocessor import (
    PredictionPostprocessor as RoadPredictionPostprocessor,
)
from utils.object_detection.prediction_postprocessor import (
    PredictionPostprocessor as ObjectDetectionPredictionPostprocessor,
)
from utils.input_reconstruction.metrics import SSIM
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
    F1Score,
)
from utils.object_detection.metrics import mAP
from utils.object_detection_3d.metrics import mAP_BEV, mAP_3D
from utils.shared.visualizer import Visualizer
from utils.input_reconstruction.visualizer import (
    Visualizer as InputReconstructionVisualizer,
)
from utils.object_detection.visualizer import Visualizer as ObjectDetectionVisualizer
from utils.depth.visualizer import Visualizer as DepthVisualizer
from utils.road_detection.visualizer import Visualizer as RoadVisualizer
from utils.shared.dict_utils import list_of_dict_to_dict

NUM_CLASSES = 4
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


############################## CONFIG UTILS ###############
def read_configs_txt_file(filepath: pathlib.Path) -> list[str]:
    with open(filepath) as f:
        return [line.strip() for line in f.readlines()]


def find_configs_yaml_file(
    configs_directory: pathlib.Path, config_numbers: list[str]
) -> pathlib.Path:
    wanted_config_number = config_numbers[0]
    if configs_directory.is_dir():
        for yaml_file in configs_directory.iterdir():
            curr_config_number = yaml_file.parts[-1].split("_")[0]
            if wanted_config_number == curr_config_number:
                return yaml_file


def save_yaml_file(path: pathlib.Path, yaml_file: dict) -> None:
    yaml_filename = yaml_file["name"]
    folder_path = path / yaml_filename
    folder_path.mkdir(parents=True, exist_ok=True)
    with open(folder_path / f"{yaml_filename}.yaml", "w") as file:
        yaml.dump(yaml_file, file)


def load_yaml_file(yaml_file: pathlib.Path) -> dict:
    with open(yaml_file.absolute()) as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as exception:
            print(exception)


def write_configs_txt_file(path_to_file: pathlib.Path, lines: list[str]) -> None:
    with open(path_to_file, "w") as f:
        f.writelines(f"{line}\n" for line in lines)


############################## CONFIG UTILS ##############################
def prepare_save_directories(args: dict, subfolder_name="train") -> None:
    save_dir = pathlib.Path(
        os.path.join(args["save_path"], args["name"], subfolder_name)
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    return save_dir


def configure_dataset(dataset_configs: dict[str, str | list], mode: str) -> Dataset:
    dataset_dict = {
        KittiDataset.__name__: KittiDataset,
        NuScenesNuImagesDataset.__name__: NuScenesNuImagesDataset,
    }
    if dataset_configs["dataset_name"] == KittiDataset.__name__:
        dataset_configs["task_sample_list_path"] = dataset_configs.pop(
            f"task_sample_list_path_{mode}"
        )
    return dataset_dict[dataset_configs["dataset_name"]](**dataset_configs)


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
    input_reconstruction_decoder = _configure_decoder(
        model_configs["input_reconstruction_decoder"]
    )
    depth_decoder = _configure_decoder(model_configs.get("depth_decoder", None))
    road_detection_decoder = _configure_decoder(
        model_configs.get("road_detection_decoder", None)
    )

    if road_detection_decoder:
        road_detection_decoder = road_detection_decoder.to(device)

    necks_and_heads = _configure_necks_and_heads(
        model_configs.get("heads_and_necks", None), device
    )
    model = MultiTaskNetwork(
        encoder=encoder,
        input_reconstruction_decoder=input_reconstruction_decoder,
        depth_decoder=depth_decoder,
        road_detection_decoder=road_detection_decoder,
        heads_and_necks=necks_and_heads,
    )
    _load_model_weights(model, model_configs)
    print_model_size(model)

    return model


def _configure_encoder(encoder_configs: dict) -> nn.Module:
    encoder_dict = {f"{ResNet.__name__}18": ResNet18}

    return encoder_dict[encoder_configs["name"]]()


def _configure_decoder(decoder_configs: dict) -> nn.Module:
    decoder_dict = {
        UnetInputReconstructionDecoder.__name__: UnetInputReconstructionDecoder,
        UnetDepthDecoder.__name__: UnetDepthDecoder,
        UnetRoadDetectionDecoder.__name__: UnetRoadDetectionDecoder,
    }

    if not decoder_configs:
        return None

    return decoder_dict[decoder_configs["name"]](**decoder_configs)


def _configure_necks_and_heads(
    necks_and_heads_configs: dict, device: torch.device
) -> nn.Module:
    necks_and_heads_dict = {FPNFasterRCNN.__name__: FPNFasterRCNN}

    if not necks_and_heads_configs:
        return None

    return FPNFasterRCNN(necks_and_heads_configs)


def _load_model_weights(model: MultiTaskNetwork, model_configs: dict):
    """
    Loads model weights using regular expressions.
    Inside the .yaml files under pretrained_regex we define all the model weights we wish
    to load from earlier ran models.

    Args:
        - model_configs: Model configuration information.
    """
    weights_file_path = model_configs.pop("weights_file_path", None)
    if not weights_file_path:
        return

    model_pretrained_dict = torch.load(weights_file_path)
    model_dict = model.state_dict()
    for model_part, part_info in model_configs.items():
        pretrained_regex_list = part_info.get("pretrained_regex", None)
        if not pretrained_regex_list:
            # If the submodule is not pretrained
            continue
        for pretrained_regex in pretrained_regex_list:
            matched_layer_names = [
                re.search(f"{model_part}.{pretrained_regex}", layer_name).string
                for layer_name in model_dict.keys()
                if re.search(f"{model_part}.{pretrained_regex}", layer_name)
            ]
            print(f"Matched layer names for {model_part}: {matched_layer_names}")
            model_dict.update(
                {
                    matched_layer_name: model_pretrained_dict[matched_layer_name]
                    for matched_layer_name in matched_layer_names
                }
            )

    model.load_state_dict(model_dict)


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
        "input_reconstruction_decoder": model.input_reconstruction_decoder,
        "depth_decoder": model.depth_decoder,
        "road_detection_decoder": getattr(model, "road_detection_decoder", None),
    }

    for submodule_name, submodule in submodules_dict.items():
        submodule_config = model_configs.get(submodule_name, {})

        if isinstance(submodule_config, list):
            submodule_config = list_of_dict_to_dict(submodule_config)

        if submodule:
            if submodule_config.get("freeze_epoch") == epoch:
                freeze_params(submodule, freeze=True)
            if submodule_config.get("unfreeze_epoch") == epoch:
                freeze_params(submodule, freeze=False)

    heads_and_necks_config = model_configs.get("heads_and_necks", {})
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
        # input reconstruction losses
        MSE.__name__: MSE,
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
        if loss_configs[task]:
            # quick fix to autoencoder structure
            for loss_name_args_dict in loss_configs[task]:
                for loss_name, args_list in loss_name_args_dict.items():
                    task_losses[task].append(
                        loss_dict[loss_name](**list_of_dict_to_dict(args_list))
                    )

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
        TaskEnum.input: InputPredictionPostProcessor,
        TaskEnum.depth: DepthPredictionPostProcessor,
        TaskEnum.road_detection: RoadPredictionPostprocessor,
        TaskEnum.object_detection: ObjectDetectionPredictionPostprocessor,
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
        # input reconstruction
        MSE.__name__: MSE,
        SSIM.__name__: SSIM,
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
        F1Score.__name__: F1Score,
        # object detection 2D metrics
        mAP.__name__: mAP,
        # object detection 3D metrics
        mAP_BEV.__name__: mAP_BEV,
        mAP_3D.__name__: mAP_3D,
    }
    task_metrics = {task: [] for task in metric_configs.keys()}
    for task in metric_configs.keys():
        if metric_configs[task]:
            # quick fix to autoencoder structure
            for metric_name in metric_configs[task]:
                task_metrics[task].append(metrics_dict[metric_name]())

    return MultiTaskMetrics(task_metrics)


def configure_visualizers(
    configs: dict, save_dir: pathlib.Path, epoch: int
) -> Visualizer:
    visualizer_configs = (
        configs["dataset"]["task_paths"]
        if configs["dataset"].get("task_paths", None)
        else configs["dataset"]["tasks"]
    )
    save_path = save_dir / "images"

    visualizers_dict = {
        InputReconstructionVisualizer.task: InputReconstructionVisualizer,
        ObjectDetectionVisualizer.task: ObjectDetectionVisualizer,
        DepthVisualizer.task: DepthVisualizer,
        RoadVisualizer.task: RoadVisualizer,
    }

    return Visualizer(
        {
            task: visualizers_dict[task]()
            for task in visualizer_configs
            if task in visualizers_dict
        },
        save_dir=save_path,
        epoch=epoch,
    )


############################## SHARED UTILS ##############################
def move_data_to_gpu(
    data: dict[str, torch.Tensor | dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    for task, value in data.items():
        if isinstance(value, torch.Tensor):
            data[task] = value.to(device="cuda")
        elif isinstance(value, dict):
            for subkey, subvalue in value.items():
                if isinstance(subvalue, torch.Tensor):
                    data[task][subkey] = subvalue.to("cuda")

    return data


def remove_dummy_ground_truth(
    data: dict[str, torch.Tensor | dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    filtered_data = defaultdict()
    for task, value in data.items():
        if isinstance(value, torch.Tensor):
            valid = _check_ground_truth_validity(
                data=filtered_data, task=task, ground_truth=value
            )
        elif isinstance(value, dict):
            subdict = {}
            for subkey, subvalue in value.items():
                valid = _check_ground_truth_validity(
                    data=subdict, task=subkey, ground_truth=subvalue
                )
                if not valid:
                    break
            if valid:
                filtered_data[task] = subdict

    return filtered_data


def _check_ground_truth_validity(
    data: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    task: str,
    ground_truth: torch.Tensor,
) -> None:
    if not torch.any(torch.isnan(ground_truth)):
        data[task] = ground_truth
        return True

    return False
