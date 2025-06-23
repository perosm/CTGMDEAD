import pathlib

import numpy as np
import cv2
import torch
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F

from utils.shared.enums import TaskEnum

NUSCENES_H, NUSCENES_W = 900, 1600
NEW_H, NEW_W = 896, 1600
IMAGE_SIZE = (NEW_H, NEW_W)
DELTA_PRINCIPAL_POINT_X = (NUSCENES_W - NEW_W) / 2
DELTA_PRINCIPAL_POINT_Y = NUSCENES_H - NEW_H
TASK_FOLDER_NAME_MAPPING = {
    TaskEnum.input: "image_2",
    TaskEnum.depth: "velodyne",
    TaskEnum.object_detection: "label_2",
    TaskEnum.road_detection: "DummyFilePath",  # We use official NuImages implementation to load segmentation data
}


TASK_FILE_EXTENSION = {
    TaskEnum.input: "png",
    TaskEnum.depth: "bin",
    TaskEnum.object_detection: "txt",
    TaskEnum.road_detection: "png",
}


class CropImage(object):
    def __init__(self) -> None:
        """
        Mimics https://pytorch.org/vision/main/generated/torchvision.transforms.functional.crop.html
        """
        self.top = NUSCENES_H - NEW_H
        self.left = (NUSCENES_W - NEW_W) // 2
        self.height = NEW_H
        self.width = NEW_W

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return F.crop(img, self.top, self.left, self.height, self.width)


TASK_TRANSFORMS = {"ToTensor": transforms.ToTensor(), "Crop": CropImage()}


FLAT_DRIVEABLE_SURFACE_INDEX = 24


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
