import pathlib

import numpy as np
import cv2
import torch
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F

from utils.shared.enums import TaskEnum

NUSCENES_H, NUSCENES_W = 900, 1600
IMAGE_SIZE = (NUSCENES_H, NUSCENES_W)
# NEW_H, NEW_W = 256, 1184
TASK_FOLDER_NAME_MAPPING = {
    TaskEnum.input: "image_2",
    TaskEnum.depth: "velodyne",
    TaskEnum.object_detection: "label_2",
    TaskEnum.road_detection: "...",
}


TASK_FILE_EXTENSION = {
    TaskEnum.input: "png",
    TaskEnum.depth: "bin",
    TaskEnum.object_detection: "txt",
    TaskEnum.road_detection: "...",
}

TASK_TRANSFORMS = {"ToTensor": transforms.ToTensor()}


############################## DATA LOAD UTILS ##############################
def input_load_util():
    """
    Used to return the function that will load the input data.
    """

    def func(png_file_path):
        return torchvision.io.decode_image(png_file_path).to(torch.float32)

    return func


def depth_load_util():
    """
    Used to return the function that will load the depth data.
    """

    def func(png_file_path: pathlib.Path) -> torch.Tensor:
        if not png_file_path.exists():
            return torch.fill_(torch.empty((3, NUSCENES_H, NUSCENES_W)), torch.nan)

        return torch.from_numpy(
            cv2.imread(png_file_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            / 256.0  # maybe https://pytorch.org/vision/master/generated/torchvision.io.decode_image.html#torchvision.io.decode_image?
        ).unsqueeze(0)

    return func


SEMANTIC_SEGMENTATION_MAPPING = {
    "sky": torch.Tensor([128.0, 128.0, 128.0]),
    "building": torch.Tensor([128.0, 0.0, 0.0]),
    "road": torch.Tensor([128.0, 64.0, 128.0]),
    "sidewalk": torch.Tensor([0.0, 0.0, 192.0]),
    "fence": torch.Tensor([64.0, 64.0, 128.0]),
    "vegetation": torch.Tensor([128.0, 128.0, 0.0]),
    "pole": torch.Tensor([192.0, 192.0, 128.0]),
    "car": torch.Tensor([64.0, 0.0, 128.0]),
    "sign": torch.Tensor([192.0, 128.0, 128.0]),
    "pedestrian": torch.Tensor([64.0, 64.0, 0.0]),
    "cyclist": torch.Tensor([0.0, 128.0, 192.0]),
    "ignore": torch.Tensor([0.0, 0.0, 0.0]),
}


def road_detection_load_util():
    """
    Used to return the function that will load road detection ground truth.

    Note: Due to labelling of kitti road detection data we use semantic segmentation data
    and convert labels so we just use the road labeling.
    """

    def func(png_file_path: pathlib.Path) -> torch.Tensor:
        if not png_file_path.exists():
            return torch.fill_(torch.empty((1, KITTI_H, KITTI_W)), torch.nan)

        ground_truth = torchvision.io.decode_image(png_file_path, mode="RGB").to(
            torch.float32
        )
        mask = (
            torch.where(
                (
                    ground_truth == SEMANTIC_SEGMENTATION_MAPPING["road"].view(3, 1, 1)
                ).all(dim=0),
                1,
                0,
            )
            .unsqueeze(0)
            .to(dtype=torch.float32)
        )
        return mask

    return func


# type (1), truncated (1), occluded (1), alpha (1), bbox (4), dimension (3), location (3), rotation_y (1)
OBJDET_LABEL_SHAPE = 15
OBJDET_CLASS_MAPPING = {
    # 0 reserved for no object
    # NuScenes
    "car": 1,
    "motorcycle": 2,
    "bicycle": 2,
    "pedestrian": 2,
    "bus": 3,
    "truck": 3,
    "trailer": 3,
    "construction_vehicle": 3,
    "barrier": -1,
    "traffic_cone": -1,
    # Kitti
    # "Car": 1,
    # "Van": 1,
    # "Truck": 1,
    # "Pedestrian": 2,
    # "Person_sitting": 2,
    # "Cyclist": 2,
    # "Tram": 3,
    # "Misc": -1,
    # "DontCare": -1,
}
