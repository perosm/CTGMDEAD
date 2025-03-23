import re

import numpy as np
import cv2
import torch
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F

KITTI_H, KITTI_W = 375, 1242
NEW_H, NEW_W = 256, 1184


############################## DATA LOAD UTILS ##############################
def input_load_util():
    """
    Used to return the function that will load the input data.
    """

    def func(png_file_path):
        return torchvision.io.read_image(png_file_path).to(torch.float32)

    return func


def depth_load_util():
    """
    Used to return the function that will load the depth data.
    """

    def func(png_file_path):
        return torch.from_numpy(
            cv2.imread(png_file_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            / 256.0  # maybe https://pytorch.org/vision/master/generated/torchvision.io.decode_image.html#torchvision.io.decode_image?
        ).unsqueeze(0)

    return func


# TODO: finish for objdet
OBJDET_LABEL_SHAPE = 5  # (type, x1, y1, x2, y2)
objdet_class_mapping = {
    "Car": 0,
    "Van": 1,
    "Truck": 2,
    "Pedestrian": 3,
    "Person_sitting": 4,
    "Cyclist": 5,
    "Tram": 6,
    "Misc": 7,
    "DontCare": 8,
}


def objdet_load_util():
    """
    Used to return the function that will load the object detection labels accordingly.
    """

    def func(txt_file_path):
        with open(txt_file_path, "r") as file:
            lines = file.readlines()
        NUM_DETECTIONS = len(lines)
        y = torch.zeros(shape=(NUM_DETECTIONS, OBJDET_LABEL_SHAPE))
        for i in range(len(lines)):
            elements = lines[i].split(" ")
            type = objdet_class_mapping[elements[0]]
            left = float(elements[4])
            top = float(elements[5])
            right = float(elements[6])
            bottom = float(elements[7])
            y[i] = torch.Tensor([type, left, top, right, bottom])

        return y

    return func


def load_utils(tasks: list[str]) -> dict:
    """
    For given tasks returns utility functions to accordingly load the data.
    """
    task_load_type = {
        "input": input_load_util(),
        "depth": depth_load_util(),
        "objdet": objdet_load_util(),
    }
    return {task: task_load_type[task] for task in tasks}


############################## TRANSFORM UTILS ##############################
class CropImage(object):
    def __init__(self) -> None:
        """
        Mimics https://pytorch.org/vision/main/generated/torchvision.transforms.functional.crop.html
        """
        self.top = KITTI_H - NEW_H
        self.left = (KITTI_W - NEW_W) // 2
        self.height = NEW_H
        self.width = NEW_W

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return F.crop(img, self.top, self.left, self.height, self.width)


def task_tranform_mapping(task_transforms_str: dict[str, str]) -> dict[str, list]:
    """
    Pairs strings from .yaml file to torch transforms accordingly.

    Args:
        - task_transforms_str: dictionary which pairs each task to its corresponding data transforms.
    """
    transforms_dict = {
        "Crop": CropImage(),
        "ToTensor": transforms.ToTensor(),
    }
    task_transforms = {task: [] for task in task_transforms_str.keys()}
    for task in task_transforms_str.keys():
        for transform_str in task_transforms_str[task]:
            task_transforms[task].append(transforms_dict[transform_str])

    return task_transforms
