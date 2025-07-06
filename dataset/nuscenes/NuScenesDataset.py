import os
import pathlib

import torch
import numpy as np
import cv2
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt

from dataset.nuscenes.nuscenes_devkit.nuimages.nuimages import NuImages
from utils.shared.enums import TaskEnum, ObjectDetectionEnum
from PIL import Image
from dataset.nuscenes import dataset_utils as NuScenesNuImagesUtils
from utils.object_detection_3d import utils as ObjectDetection3DUtils
from torchvision.ops import clip_boxes_to_image


class NuScenesNuImagesDataset(Dataset):

    def __init__(
        self,
        tasks: list[str] = [
            TaskEnum.input,
            TaskEnum.depth,
            TaskEnum.object_detection,
            TaskEnum.road_detection,
        ],
        version: str = "v1.0-train",
        nuscenes_kitti_dataroot: str = "./data/nuscenes_kitti/train",
        nuimages_dataroot: str = "./data/nuscenes/nuimages",
        num_samples_train: int = 6000,
        num_samples_val: int = 500,
        mode: str = "train",
        **kwargs,
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
            num_samples_train: Integer representing the number of samples used for training * 2.
            num_samples_val: Integer representing the number of samples used for validation * 2.
            mode: String representing mode (i.e. train or val)
        """
        super().__init__()
        self.tasks = tasks
        self.nuscenes_kitti_root = pathlib.Path(nuscenes_kitti_dataroot)
        self.nuscenes_task_dataroot = self._fetch_task_dataroot()
        self.nuimages_dataroot = pathlib.Path(nuimages_dataroot)
        self.nuimages = NuImages(
            version=version, dataroot=nuimages_dataroot, verbose=False, lazy=True
        )  # TODO: change verbose=False
        self.num_samples_train = num_samples_train
        self.num_samples_val = num_samples_val
        self.mode = mode

        self.nuscenes_sample_list = self._read_nuscenes_sample_list()
        self.nuimages_sample_list = self._read_nuimages_sample_list()
        self.sample_list = self.nuscenes_sample_list + self.nuimages_sample_list

    def _fetch_task_dataroot(self) -> dict[str, pathlib.Path]:
        task_dataroot = {}
        for task in self.tasks:
            folder_name = NuScenesNuImagesUtils.TASK_FOLDER_NAME_MAPPING.get(task, None)
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

        return (
            sample_list[: self.num_samples_train]
            if self.mode == "train"
            else sample_list[: self.num_samples_val]
        )

    def _read_nuimages_sample_list(self) -> list[str]:
        """
        Sample list for NuImages is shared for the tasks of:
            - input
            - road detection

        Returns:
            NuScenes sample list as a list of scene tokens.
        """
        sample_list = []

        for surface_ann in self.nuimages.surface_ann:
            sample_data_token = surface_ann["sample_data_token"]
            category_token = surface_ann["category_token"]
            category = self.nuimages.get("category", category_token)
            category_name = category["name"]
            if category_name == "flat.driveable_surface":
                sample_data = self.nuimages.get("sample_data", sample_data_token)
                sample_list.append(sample_data["sample_token"])

        return (
            sample_list[: self.num_samples_train]
            if self.mode == "train"
            else sample_list[: self.num_samples_val]
        )

    @staticmethod
    def _read_input(filepath: pathlib.Path) -> torch.Tensor:
        image = Image.open(fp=filepath)
        # plt.imshow(image)
        # plt.title("input_image")
        return NuScenesNuImagesUtils.TASK_TRANSFORMS["Crop"](
            NuScenesNuImagesUtils.TASK_TRANSFORMS["ToTensor"](image)
        )

    @staticmethod
    def _read_depth(filepath: pathlib.Path) -> torch.Tensor:
        pcl = np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)

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
        num_points = pcl.shape[0]
        pcl[:, :3] = (
            transformation_matrix
            @ np.concatenate([pcl[:, :3], np.ones([num_points, 1])], axis=1).T
        ).T[:, :3]

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
        pcl = pcl[np.where(pcl[:, 2] >= 0)[0]]  # Points with positive depth
        depth = pcl[:, 2]
        points_2d = ObjectDetection3DUtils.project_points_to_image_numpy(
            points_3d=pcl[:, :3], projection_matrix=rectification_projection_matrix
        )
        image = np.zeros((*NuScenesNuImagesUtils.IMAGE_SIZE, 1))
        # Points with x in range [0, IMAGE_WIDTH)
        keep = np.where(points_2d[:, 0] >= 0, True, False)
        keep = keep & np.where(
            points_2d[:, 0] < NuScenesNuImagesUtils.IMAGE_SIZE[1], True, False
        )

        # Points with y in range [0, IMAGE_HEIGHT)
        keep = keep & np.where(points_2d[:, 1] >= 0, True, False)
        keep = keep & np.where(
            points_2d[:, 1] < NuScenesNuImagesUtils.IMAGE_SIZE[0], True, False
        )
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
        return NuScenesNuImagesUtils.TASK_TRANSFORMS["Crop"](
            torch.from_numpy(image).permute(2, 0, 1)
        ).to(torch.float32)

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
            return torch.fill_(
                torch.empty((1, NuScenesNuImagesUtils.OBJDET_LABEL_SHAPE)), torch.nan
            )

        with open(filepath, "r") as file:
            lines = [line.strip().split(" ") for line in file.readlines()]

        objects_to_keep = []
        NUM_DETECTIONS = len(lines)
        gt = np.empty(
            shape=(NUM_DETECTIONS, NuScenesNuImagesUtils.OBJDET_LABEL_SHAPE),
            dtype=np.float32,
        )
        for object_index, object_info in enumerate(lines):
            class_num = NuScenesNuImagesUtils.OBJDET_CLASS_MAPPING[object_info[0]]
            if class_num == -1:
                continue

            gt[object_index, ObjectDetectionEnum.object_class] = (
                NuScenesNuImagesUtils.OBJDET_CLASS_MAPPING[object_info[0]]
            )  # type
            gt[object_index, ObjectDetectionEnum.truncated] = float(
                object_info[1]
            )  # truncated flag
            gt[object_index, ObjectDetectionEnum.occluded] = float(
                object_info[2]
            )  # occluded flag
            gt[object_index, ObjectDetectionEnum.alpha] = float(object_info[3])  # alpha
            gt[
                object_index,
                ObjectDetectionEnum.box_2d_left : ObjectDetectionEnum.box_2d_bottom + 1,
            ] = [
                float(image_coord)
                for image_coord in object_info[4:8]  # left, top, right, bottom
            ]
            gt[object_index, ObjectDetectionEnum.box_2d_left] -= (
                NuScenesNuImagesUtils.NUSCENES_W - NuScenesNuImagesUtils.NEW_W
            ) / 2  # left
            gt[object_index, ObjectDetectionEnum.box_2d_top] -= (
                NuScenesNuImagesUtils.NUSCENES_H - NuScenesNuImagesUtils.NEW_H
            )  # top
            gt[object_index, ObjectDetectionEnum.box_2d_right] -= (
                NuScenesNuImagesUtils.NUSCENES_W - NuScenesNuImagesUtils.NEW_W
            ) / 2  # right
            gt[object_index, ObjectDetectionEnum.box_2d_bottom] -= (
                NuScenesNuImagesUtils.NUSCENES_H - NuScenesNuImagesUtils.NEW_H
            )  # bottom
            gt[
                object_index,
                ObjectDetectionEnum.height : ObjectDetectionEnum.rotation_y + 1,
            ] = [float(world_coord) for world_coord in object_info[8:15]]
            objects_to_keep.append(object_index)

        gt_objects = torch.from_numpy(gt[objects_to_keep])
        gt_objects[
            :, ObjectDetectionEnum.box_2d_left : ObjectDetectionEnum.box_2d_bottom + 1
        ] = clip_boxes_to_image(
            gt_objects[
                :,
                ObjectDetectionEnum.box_2d_left : ObjectDetectionEnum.box_2d_bottom + 1,
            ],
            size=NuScenesNuImagesUtils.IMAGE_SIZE,
        )
        return torch.from_numpy(gt[objects_to_keep])

    def __len__(self) -> int:
        return len(self.sample_list)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        data = {}
        sample_token = self.sample_list[index]
        for task in self.tasks:
            filepath = (
                self.nuscenes_task_dataroot[task]
                / f"{sample_token}.{NuScenesNuImagesUtils.TASK_FILE_EXTENSION[task]}"
            )
            if filepath.exists():  # depth, object_detection
                #################### NuScenes dataset ####################
                if task == TaskEnum.input:
                    data[task] = NuScenesNuImagesDataset._read_input(filepath=filepath)
                    data["projection_matrix"] = torch.from_numpy(
                        NuScenesNuImagesDataset._read_transformation_matrix(
                            filepath=str(filepath)
                            .replace("image_2", "calib")
                            .replace(".png", ".txt"),
                            matrix_name="P2",
                        ).reshape(3, 4)
                    )
                elif task == TaskEnum.depth:
                    data[task] = NuScenesNuImagesDataset._read_depth(filepath=filepath)
                elif task == TaskEnum.object_detection:
                    object_detection_gt = (
                        NuScenesNuImagesDataset._read_object_detection(
                            filepath=filepath
                        )
                    )
                    projection_matrix = torch.from_numpy(
                        NuScenesNuImagesDataset._read_transformation_matrix(
                            filepath=str(filepath).replace("label_2", "calib"),
                            matrix_name="P2",
                        ).reshape(3, 4)
                    )
                    projection_matrix[
                        0, 2
                    ] -= NuScenesNuImagesUtils.DELTA_PRINCIPAL_POINT_X
                    projection_matrix[
                        1, 2
                    ] -= NuScenesNuImagesUtils.DELTA_PRINCIPAL_POINT_Y
                    data[task] = {
                        "gt_info": object_detection_gt,
                        "projection_matrix": projection_matrix,
                    }
            elif (
                not filepath.exists() and sample_token in self.nuimages_sample_list
            ):  # road_detection
                #################### NuImages dataset ####################
                sample = self.nuimages.get("sample", sample_token)
                key_camera_token = sample["key_camera_token"]
                sample_data = self.nuimages.get("sample_data", key_camera_token)
                calibrated_sensor_data = self.nuimages.get(
                    "calibrated_sensor", sample_data["calibrated_sensor_token"]
                )
                if task == TaskEnum.input:
                    filepath = os.path.join(
                        self.nuimages.dataroot, sample_data["filename"]
                    )
                    data[task] = NuScenesNuImagesDataset._read_input(filepath)
                    projection_matrix = np.eye(4)
                    projection_matrix[:3, :3] = np.array(
                        calibrated_sensor_data["camera_intrinsic"]
                    ).reshape(3, 3)
                    data["projection_matrix"] = torch.from_numpy(projection_matrix)[
                        :3, :
                    ]
                if task == TaskEnum.road_detection:
                    semantic_mask, _ = self.nuimages.get_segmentation(key_camera_token)
                    road_mask = np.where(
                        semantic_mask
                        == NuScenesNuImagesUtils.FLAT_DRIVEABLE_SURFACE_INDEX,
                        1,
                        0,
                    )
                    # plt.imshow(road_mask)
                    # plt.title("road_detection")
                    # plt.show()
                    data[task] = NuScenesNuImagesUtils.TASK_TRANSFORMS["Crop"](
                        torch.from_numpy(road_mask).to(torch.float32)
                    ).unsqueeze(0)
        return data


if __name__ == "__main__":
    from tqdm import tqdm

    dataset = NuScenesNuImagesDataset()

    for sample_idx, data in enumerate(tqdm(dataset, "Loading samples...")):
        print("Sample idx: ", data[TaskEnum.object_detection]["gt_info"].shape)
        if data[TaskEnum.object_detection]["gt_info"].shape[0] == 0:
            plt.imshow(data[TaskEnum.input].permute(1, 2, 0).numpy())
            plt.show()
