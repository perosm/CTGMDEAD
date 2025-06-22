import os

import torch
import numpy as np
import pathlib
from torch.utils.data.dataset import Dataset
from dataset.nuscenes.nuscenes_devkit.nuscenes.nuscenes import NuScenes
from dataset.nuscenes.nuscenes_devkit.nuimages.nuimages import NuImages
import matplotlib.pyplot as plt

from utils.shared.enums import TaskEnum
from PIL import Image
from torchvision import transforms
from dataset.nuscenes.dataset_utils import (
    TASK_FOLDER_NAME_MAPPING,
    TASK_FILE_EXTENSION,
    OBJDET_LABEL_SHAPE,
    OBJDET_CLASS_MAPPING,
    TASK_TRANSFORMS,
    IMAGE_SIZE,
)
from dataset.nuscenes.nuscenes_devkit.nuscenes.utils.data_classes import LidarPointCloud
from utils.object_detection_3d.utils import project_points_to_image_numpy
from pyquaternion import Quaternion


class NuScenesNuImagesDataset(Dataset):

    def __init__(
        self,
        tasks: list[str] = [
            TaskEnum.input,
            TaskEnum.depth,
            TaskEnum.object_detection,
        ],  # road detection
        version: str = "v1.0-mini",
        nuscenes_kitti_dataroot: str = "./data/nuscenes_kitti/mini_train",
        nuimages_dataroot: str = "./data/nuscenes/nuimages",
    ):
        """
        Dataset loader used for NuScenes and NuImages dataset.
        NuScenes data is exported to kitti format beforehand.
        NuImages data uses the official dataloader.

        Args:
            tasks: List of tasks which we train our network.
            version: Dataset version (used for NuImages loader instancing).
            nuscenes_kitti_dataroot: Dataset root of images, calibrations, object detection labels and velodyne pointcloud.
            nuimages_dataroot: Root folder where nuimages data is stored.
        """
        super().__init__()
        self.tasks = tasks
        self.nuscenes_kitti_root = pathlib.Path(nuscenes_kitti_dataroot)
        self.nuimages_task_dataroot = self._fetch_task_dataroot()
        self.nuimages_dataroot = pathlib.Path(nuimages_dataroot)
        self.nuimages = NuImages(
            version=version, dataroot=nuimages_dataroot, verbose=True, lazy=True
        )  # TODO: change verbose=False

        self.nuscenes_sample_list = self._read_nuscenes_sample_list()
        self.nuimages_sample_list = self._read_nuimages_sample_list()
        self.sample_list = self.nuscenes_sample_list + self.nuimages_sample_list

    def _fetch_task_dataroot(self) -> dict[str, pathlib.Path]:
        task_dataroot = {}
        for task in self.tasks:
            folder_name = TASK_FOLDER_NAME_MAPPING.get(task, None)
            if folder_name:
                task_dataroot[task] = self.nuscenes_kitti_root / folder_name

        return task_dataroot

    def _read_nuscenes_sample_list(self) -> list[str]:
        """
        Sample list for NuScenes is shared for the tasks of:
            - input
            - depth
            - object_detection

        Returns:
            NuScenes sample list as a list of scene tokens.
        """
        sample_list = sorted(
            [
                token_extension.split(".")[
                    0
                ]  # remove ".png" extension part, take just the token
                for token_extension in os.listdir(self.nuscenes_kitti_root / "image_2")
            ]
        )

        return sample_list

    def _read_nuimages_sample_list(self) -> list[str]:
        """
        Sample list for NuImages is shared for the tasks of:
            - object_detection_2d TODO: Should I ignore this?
            - road detection

        Returns:
            NuScenes sample list as a list of scene tokens.
        """
        sample_list = []
        for sample_record in self.nuimages.sample:
            sample_list.append(sample_record["token"])

        return sample_list

    @staticmethod
    def _read_input(filepath: pathlib.Path) -> torch.Tensor:
        image = Image.open(fp=filepath)

        return TASK_TRANSFORMS["ToTensor"](image)

    @staticmethod
    def _read_depth(filepath: pathlib.Path) -> torch.Tensor:
        kitti_to_nu_lidar_inv = Quaternion(axis=(0, 0, 1), angle=np.pi / 2).inverse
        pcl = LidarPointCloud.from_file(file_name=str(filepath))
        # pcl.rotate(kitti_to_nu_lidar_inv)  # Rotate to KITTI lidar.

        # Transform pointcloud to camera frame.
        transformation_matrices_filepath = pathlib.Path(
            str(filepath).replace("velodyne", "calib").replace("bin", "txt")
        )
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :] = (
            NuScenesNuImagesDataset._read_transformation_matrix(
                filepath=transformation_matrices_filepath, matrix_name="Tr_velo_to_cam"
            ).reshape(3, 4)
        )
        pcl.transform(transf_matrix=transformation_matrix)

        # Project points to image
        rectification_projection_matrix = np.eye(4)
        rectification_matrix = NuScenesNuImagesDataset._read_transformation_matrix(
            filepath=transformation_matrices_filepath, matrix_name="R0_rect"
        ).reshape(3, 3)
        rectification_projection_matrix[:3, :3] = rectification_matrix

        projection_matrix = NuScenesNuImagesDataset._read_transformation_matrix(
            filepath=transformation_matrices_filepath, matrix_name="P2"
        ).reshape(3, 4)
        rectification_projection_matrix = (
            projection_matrix @ rectification_projection_matrix
        )
        points = pcl.points.T  # (num_points, 4)
        points = points[np.where(points[:, 2] >= 0)[0]]  # Points with positive depth
        depth = points[:, 2]
        points_2d = project_points_to_image_numpy(
            points_3d=points[:, :3], projection_matrix=rectification_projection_matrix
        )
        image = np.zeros((*IMAGE_SIZE, 1))
        # Points with x in range [0, IMAGE_WIDTH)
        keep = np.where(points_2d[:, 0] >= 0, True, False)
        keep = keep & np.where(points_2d[:, 0] < IMAGE_SIZE[1], True, False)

        # Points with y in range [0, IMAGE_HEIGHT)
        keep = keep & np.where(points_2d[:, 1] >= 0, True, False)
        keep = keep & np.where(points_2d[:, 1] < IMAGE_SIZE[0], True, False)
        points_2d = points_2d[keep]
        depth = depth[keep]

        for depth_xy, (point_x, point_y) in zip(depth, points_2d):
            image[int(point_y), int(point_x)] = depth_xy

        # scatter_plot = plt.scatter(
        #     points_2d[:, 0],
        #     [points_2d[:, 1]],
        #     c=[depth],
        #     cmap="rainbow_r",
        #     alpha=0.5,
        #     s=2,
        # )
        # cbar = plt.colorbar(scatter_plot)
        # cbar.set_label("depth")
        # plt.show()
        return torch.from_numpy(image)  # TODO: check again if this is ok

    @staticmethod
    def _read_transformation_matrix(
        filepath: pathlib.Path, matrix_name: str
    ) -> np.ndarray:
        matrix_name_transformation = {}
        with open(filepath, "r") as file:
            lines = file.readlines()

        for line in lines:
            name, transformation_matrix = line.strip().split(": ")
            matrix_name_transformation[name] = np.array(
                [float(number) for number in transformation_matrix.split()],
                dtype=np.float32,
            )

        return matrix_name_transformation[matrix_name]

    @staticmethod
    def _read_object_detection(filepath: pathlib.Path) -> torch.Tensor:
        if not filepath.exists():
            return torch.fill_(torch.empty((1, OBJDET_LABEL_SHAPE)), torch.nan)

        with open(filepath, "r") as file:
            lines = [line.strip().split(" ") for line in file.readlines()]

        objects_to_keep = []
        NUM_DETECTIONS = len(lines)
        gt = np.empty(shape=(NUM_DETECTIONS, OBJDET_LABEL_SHAPE), dtype=np.float32)
        for object_index, object_info in enumerate(lines):
            class_num = OBJDET_CLASS_MAPPING[object_info[0]]
            if class_num == -1:
                continue

            gt[object_index, 0] = OBJDET_CLASS_MAPPING[object_info[0]]  # type
            gt[object_index, 1] = float(object_info[1])  # truncated flag
            gt[object_index, 2] = float(object_info[2])  # occluded flag
            gt[object_index, 3] = float(object_info[3])  # alpha
            gt[object_index, 4:8] = [
                float(image_coord)
                for image_coord in object_info[4:8]  # left, top, right, bottom
            ]
            gt[object_index, 8:15] = [
                float(world_coord) for world_coord in object_info[8:15]
            ]
            objects_to_keep.append(object_index)

        return torch.from_numpy(gt[objects_to_keep])

    @staticmethod
    def _read_road_detection(filepath: pathlib.Path) -> torch.Tensor:
        pass

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        data = {}
        sample_token = self.sample_list[index]
        for task in self.tasks:
            filepath = (
                self.nuimages_task_dataroot[task]
                / f"{sample_token}.{TASK_FILE_EXTENSION[task]}"
            )
            if task == TaskEnum.input:
                data[task] = NuScenesNuImagesDataset._read_input(filepath=filepath)
            elif task == TaskEnum.depth:
                data[task] = NuScenesNuImagesDataset._read_depth(filepath=filepath)
            elif task == TaskEnum.object_detection:
                data[task] = NuScenesNuImagesDataset._read_object_detection(
                    filepath=filepath
                )
            elif task == TaskEnum.road_detection:
                data[task] = ...

        return data


if __name__ == "__main__":
    dataset = NuScenesNuImagesDataset()

    sample = next(iter(dataset))
