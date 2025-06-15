import torch
import pathlib
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import dataset.kitti.dataset_utils as KITTIutils
from utils.shared.enums import TaskEnum
from collections import defaultdict

DELTA_PRINCIPAL_POINT_X = (KITTIutils.KITTI_W - KITTIutils.NEW_W) / 2
DELTA_PRINCIPAL_POINT_Y = KITTIutils.KITTI_H - KITTIutils.NEW_H


class KittiDataset(Dataset):
    DATE_INDEX = -5
    UNIQUE_PATH_PARTS_NUMBER = -5

    def __init__(
        self,
        task_paths: dict[str, str],
        task_transform: dict[str, list],
        task_sample_list_path: dict[str, str] = None,
        camera: str = "image_02",
        co_train: bool = False,
    ) -> None:
        """
        Args:
         - task_transforms: dictionary of transforms for each of the network tasks.
         - camera: string which represents left camera ('02') or right camera ('03')
         - paths: paths to input data and ground truth for neural networks tasks (e.g. object detection, depth estimation...)
                  paired with the path to the folder where the ground truth for each of the tasks is stored.
         - task_sample_list: List of samples used for that task (used for filtering).
        """
        self.co_train = co_train
        self.camera = camera
        self.camera_index = int(camera[-1])
        self.ground_truth_path_to_projection_matrices = {}
        self.paths_dict = defaultdict(list)
        self.task_sample_list = self._read_sample_lists(task_sample_list_path)
        self.task_root_dir_path = {
            task: pathlib.Path(task_paths[task]) for task in task_paths.keys()
        }
        self.task_transform = self._configure_transforms(
            KITTIutils.task_tranform_mapping(task_transform)
        )
        self._load_data_paths()
        if TaskEnum.object_detection in self.task_root_dir_path.keys():
            self._fetch_projection_matrices()
        self._filter_data_paths()
        self.load_functions = KITTIutils.load_utils(list(self.paths_dict.keys()))

    def _read_sample_lists(self, task_sample_list_path: dict[str, str]):
        task_sample_list = defaultdict(set)
        for task, sample_list_path in task_sample_list_path.items():
            with open(sample_list_path, "r") as file:
                lines = file.readlines()
                for line in lines:
                    task_sample_list[task].add(line.strip())

        return task_sample_list

    def _configure_transforms(
        self, task_transform: dict
    ) -> dict[str, transforms.Compose]:
        return {
            task: transforms.Compose(*[task_transform[task]]) for task in task_transform
        }

    def _fetch_projection_matrices(self) -> None:
        if not self.paths_dict[TaskEnum.object_detection]:
            return

        for ground_truth_objdet3d_path in self.paths_dict[TaskEnum.object_detection]:

            calibrations_path = str(ground_truth_objdet3d_path).replace(
                "ground_truth", "calibrations"
            )
            frame_id = "/".join(
                str(ground_truth_objdet3d_path).split("/")[
                    self.UNIQUE_PATH_PARTS_NUMBER :
                ]
            )
            projection_matrix = self._read_projection_matrix(calibrations_path)
            self.ground_truth_path_to_projection_matrices[frame_id] = projection_matrix

    def _read_projection_matrix(self, object_detection_calibration_path: pathlib.Path):
        with open(object_detection_calibration_path, "r") as file:
            projection_matrix = torch.Tensor(
                [
                    float(number)
                    for number in file.readlines()[self.camera_index]
                    .split(": ")[1]
                    .strip()
                    .split()
                ]
            ).reshape(3, 4)
            projection_matrix[0, 2] -= DELTA_PRINCIPAL_POINT_X
            projection_matrix[1, 2] -= DELTA_PRINCIPAL_POINT_Y

        return projection_matrix

    def load_projection_matrix(self, frame: pathlib.Path) -> torch.Tensor:
        """
        Based on the given frame loads a projection matrix.

        For object detection loads projection matrix for specific frame,
        otherwise fetches a "default" projection matrix used in kitti for that date.

        Args:
            - frame: Frame name for which a projection matrix is fetched.

        Returns:
            Projection matrix for a frame.
        """
        projection_matrix = self.ground_truth_path_to_projection_matrices.get(
            str(frame), None
        )
        if projection_matrix is not None:
            return projection_matrix

        date = frame.parts[0]
        projection_matrix_by_date = (
            self.task_root_dir_path[TaskEnum.input] / date / "calib_cam_to_cam.txt"
        )
        return self._read_projection_matrix_by_date(projection_matrix_by_date)

    def _read_projection_matrix_by_date(
        self, calibration_path: pathlib.Path
    ) -> torch.Tensor:
        info_dict = {}
        with open(calibration_path, "r") as file:
            lines = file.readlines()
            for line in lines:
                k, value = line.split(": ")
                info_dict[k] = value

        projection_matrix = torch.Tensor(
            [
                float(num)
                for num in info_dict[f"P_rect_0{self.camera_index}"].strip().split()
            ]
        ).reshape(3, 4)
        projection_matrix[0, 2] -= DELTA_PRINCIPAL_POINT_X
        projection_matrix[1, 2] -= DELTA_PRINCIPAL_POINT_Y

        return projection_matrix

    def _load_data_paths(self) -> None:
        for task in self.task_root_dir_path.keys():
            for root_path, _, files in self.task_root_dir_path[task].walk():
                if self.camera in str(root_path):  # for input and depth
                    self.paths_dict[task].extend(
                        root_path / file
                        for file in files
                        if KITTIutils.task_check_file_extension(task, file)
                    )

            self.paths_dict[task] = sorted(self.paths_dict[task])

    def _filter_data_paths(self):
        task_absolute_root_paths = {
            task: self.task_root_dir_path[task].absolute()
            for task in self.task_root_dir_path.keys()
        }
        task_extension = {
            task: self.paths_dict[task][0].suffix
            for task in self.task_root_dir_path.keys()
        }
        final_paths = set(
            "/".join(path.with_suffix("").parts[self.UNIQUE_PATH_PARTS_NUMBER :])
            for path in self.paths_dict[KITTIutils.TaskEnum.input]
        )
        if not self.co_train:
            for task, unique_data in self.task_sample_list.items():
                final_paths = final_paths.intersection(unique_data)
        else:
            # When co-training we want to have all input images available so instead
            # of intersection we use union and for paths that don't exist we generate
            # dummy ground truth and set the loss to zero so we don't backprop
            first_task_name = next(iter(self.task_sample_list))
            final_paths = self.task_sample_list[first_task_name]
            for task, unique_data in self.task_sample_list.items():
                final_paths = final_paths.union(unique_data)

        for task in task_absolute_root_paths.keys():
            self.paths_dict[task] = sorted(
                [
                    task_absolute_root_paths[task] / f"{path}{task_extension[task]}"
                    for path in final_paths
                ]
            )

        self.length = len(final_paths)

    def get_item_name(self, idx):
        task_item = dict.fromkeys(self.paths_dict.keys())
        for task in self.paths_dict.keys():
            task_item[task] = self.paths_dict[task][idx]

        return task_item

    def __len__(self) -> int:
        """Returns number of samples."""
        return self.length

    def __getitem__(self, idx):
        frame_id = pathlib.Path(
            *self.paths_dict[TaskEnum.input][idx].parts[: self.DATE_INDEX + 1]
        )
        task_item = dict.fromkeys(self.paths_dict.keys())
        for task in self.paths_dict.keys():
            task_item[task] = self.task_transform[task](
                self.load_functions[task](self.paths_dict[task][idx])
            )
        if TaskEnum.object_detection in self.paths_dict.keys():
            frame_id = pathlib.Path(
                *self.paths_dict[TaskEnum.object_detection][idx].parts[
                    self.UNIQUE_PATH_PARTS_NUMBER :
                ]
            )
            projection_matrix = self.load_projection_matrix(frame=frame_id)
            gt = task_item[TaskEnum.object_detection]
            task_item[TaskEnum.object_detection] = {
                "gt_info": gt,
                "projection_matrix": projection_matrix,
            }

        # Always loads projection matrix which will be used for prediction postprocess
        # and for visualizing OD predictions on non OD related frames
        task_item["projection_matrix"] = self.load_projection_matrix(frame=frame_id)

        return task_item
