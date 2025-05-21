import torch
import pathlib
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import dataset.kitti.dataset_utils as KITTIutils


class KittiDataset(Dataset):
    UNIQUE_PATH_PARTS_NUMBER = -5

    def __init__(
        self,
        task_paths: dict[str, str],
        task_transform: dict[str, list],
        camera: str = "image_02",
    ) -> None:
        """
        Args:
          - task_transforms: dictionary of transforms for each of the network tasks.
          - camera: string which represents left camera ('02') or right camera ('03')
          - paths: paths to input data and ground truth for neural networks tasks (e.g. object detection, depth estimation...)
                   paired with the path to the folder where the ground truth for each of the tasks is stored.
        """
        self.task_transform = self._configure_transforms(
            KITTIutils.task_tranform_mapping(task_transform)
        )
        self.task_paths = {
            task: pathlib.Path(task_paths[task]) for task in task_paths.keys()
        }
        self.paths_dict = {key: [] for key in self.task_paths.keys()}
        self.camera = camera
        self.camera_index = int(camera[-1])
        self._load_data_paths()
        if KITTIutils.TaskEnum.object_detection_3d.name in self.task_paths.keys():
            self._fetch_projection_matrices()
        self._filter_data_paths()
        self.load_functions = KITTIutils.load_utils(list(self.paths_dict.keys()))

    def _configure_transforms(
        self, task_transform: dict
    ) -> dict[str, transforms.Compose]:
        return {
            task: transforms.Compose(*[task_transform[task]]) for task in task_transform
        }

    def _fetch_projection_matrices(self) -> None:
        delta_principal_point_x = (KITTIutils.KITTI_W - KITTIutils.NEW_W) / 2
        delta_principal_point_y = KITTIutils.KITTI_H - KITTIutils.NEW_H
        if not self.paths_dict[KITTIutils.TaskEnum.object_detection_3d]:
            return

        self.ground_truth_path_to_projection_matrices = {}
        for ground_truth_objdet3d_path in self.paths_dict[
            KITTIutils.TaskEnum.object_detection_3d
        ]:

            calibrations_path = str(ground_truth_objdet3d_path).replace(
                "ground_truth", "calibrations"
            )
            frame_id = "/".join(
                str(ground_truth_objdet3d_path).split("/")[
                    self.UNIQUE_PATH_PARTS_NUMBER :
                ]
            )
            projection_matrix = self._read_projection_matrix(calibrations_path)
            projection_matrix[0, 2] -= delta_principal_point_x
            projection_matrix[1, 2] -= delta_principal_point_y
            self.ground_truth_path_to_projection_matrices[frame_id] = projection_matrix

    def _read_projection_matrix(self, object_detection_calibration_path: pathlib.Path):
        with open(object_detection_calibration_path, "r") as file:
            last_row = torch.Tensor([[0, 0, 0, 1]])
            projection_matrix = torch.vstack(
                (
                    torch.Tensor(
                        [
                            float(number)
                            for number in file.readlines()[self.camera_index]
                            .split(": ")[1]
                            .strip()
                            .split()
                        ]
                    ).reshape(3, 4),
                    last_row,
                )
            )

        return projection_matrix

    def load_projection_matrix(self, frame: str) -> torch.Tensor:
        """
        Based on the given frame loads a projection matrix.

        For object detection loads projection matrix for specific frame,
        otherwise fetches a "default" projection matrix used in kitti.
        TODO: Should I always search for a projection matrix even if for that
        particular frame one does not exist but instead fetch the projection
        matrix from a frame which is closest?

        Args:
            - frame: Frame name for which a projection matrix is fetched.

        Returns:
            Projection matrix for a frame.
        """
        projection_matrix = self.ground_truth_path_to_projection_matrices.get(
            frame, None
        )
        if projection_matrix:
            return projection_matrix

        # TODO: fetch closest projection matrix to that frame
        return

    def _load_data_paths(self) -> None:
        for task in self.task_paths.keys():
            for root_path, _, files in self.task_paths[task].walk():
                if self.camera in str(root_path):  # for input and depth
                    self.paths_dict[task].extend(
                        root_path / file
                        for file in files
                        if KITTIutils.task_check_file_extension(task, file)
                    )

            self.paths_dict[task] = sorted(self.paths_dict[task])

    def _filter_data_paths(self):
        task_root_paths = {
            task: self.task_paths[task].absolute() for task in self.task_paths.keys()
        }
        task_extension = {
            task: self.paths_dict[task][0].suffix for task in self.task_paths.keys()
        }
        task_unique_data = {task: set() for task in self.task_paths.keys()}
        for task, paths in self.paths_dict.items():
            for path in paths:
                task_unique_data[task].add(
                    "/".join(
                        path.with_suffix("").parts[self.UNIQUE_PATH_PARTS_NUMBER :]
                    )
                )

        final_paths = task_unique_data[KITTIutils.TaskEnum.input]
        for task, unique_data in task_unique_data.items():
            final_paths = final_paths.intersection(unique_data)

        for task in task_root_paths.keys():
            self.paths_dict[task] = sorted(
                [
                    task_root_paths[task] / f"{path}{task_extension[task]}"
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
        task_item = dict.fromkeys(self.paths_dict.keys())
        for task in self.paths_dict.keys():
            task_item[task] = self.task_transform[task](
                self.load_functions[task](self.paths_dict[task][idx])
            )
        if KITTIutils.TaskEnum.object_detection_3d in self.paths_dict.keys():
            frame_id = "/".join(
                self.paths_dict[KITTIutils.TaskEnum.object_detection_3d][idx].parts[
                    self.UNIQUE_PATH_PARTS_NUMBER :
                ]
            )
            projection_matrix = self.ground_truth_path_to_projection_matrices[frame_id]
            task_item.update({"projection_matrix": projection_matrix})

        return task_item
