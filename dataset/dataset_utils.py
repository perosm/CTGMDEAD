import pathlib
import enum

import numpy as np
import cv2
import torch
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F

KITTI_H, KITTI_W = 375, 1242
NEW_H, NEW_W = 256, 1184


class TaskEnum(enum.StrEnum):
    input = "input"
    depth = "depth"
    road_detection = "road_detection"
    object_detection_3d = "object_detection_3d"


def task_check_file_extension(task: str, file_path: str):
    """
    Used for filtering right files in KittiDataset.
    """
    task_extensions = {
        TaskEnum.input: [".png"],
        TaskEnum.depth: [".png"],
        TaskEnum.road_detection: [".png"],
        TaskEnum.object_detection_3d: [".txt"],
    }

    if pathlib.Path(file_path).suffix in task_extensions[task]:
        return True

    return False


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

    def func(png_file_path):
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

    def func(png_file_path) -> torch.Tensor:
        ground_truth = torchvision.io.decode_image(png_file_path, mode="RGB").to(
            torch.float32
        )
        mask = torch.where(
            (ground_truth == SEMANTIC_SEGMENTATION_MAPPING["road"].view(3, 1, 1)).all(
                dim=0
            ),
            1,
            0,
        ).unsqueeze(0)
        return mask

    return func


# TODO: finish for objdet
OBJDET_LABEL_SHAPE = 5  # (type, x1, y1, x2, y2)
OBJDET_CLASS_MAPPING = {
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


def object_detection_3d_load_util():
    """
    Used to return the function that will load the object detection labels accordingly.
    """

    def func(txt_file_path) -> torch.Tensor:
        with open(txt_file_path, "r") as file:
            lines = file.readlines()
        NUM_DETECTIONS = len(lines)
        y = torch.zeros(shape=(NUM_DETECTIONS, OBJDET_LABEL_SHAPE))
        for i in range(len(lines)):
            elements = lines[i].split(" ")
            type = OBJDET_CLASS_MAPPING[elements[0]]
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
        TaskEnum.input: input_load_util(),
        TaskEnum.depth: depth_load_util(),
        TaskEnum.road_detection: road_detection_load_util(),
        TaskEnum.object_detection_3d: object_detection_3d_load_util(),
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
