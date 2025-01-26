import os
from pathlib import Path
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from typing import Dict, List
from torch.utils.data import DataLoader
import utils.kitti.utils as KITTIutils
import sys


class KittiDataset(Dataset):

    def __init__(
        self,
        task_paths: Dict[str, str],
        task_transform: Dict[str, list],
        camera: str = "02",
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
            task: os.path.abspath(task_paths[task]) for task in task_paths.keys()
        }
        self.camera = camera

        self._load_data_paths()
        self._filter_data_paths()

    def _configure_transforms(
        self, task_transform: dict
    ) -> Dict[str, transforms.Compose]:
        return {
            task: transforms.Compose(*[task_transform[task]]) for task in task_transform
        }

    def _load_data_paths(self) -> None:
        self.paths_dict = {key: [] for key in self.task_paths.keys()}
        for key in self.task_paths.keys():
            for root_path, dirs, files in os.walk(self.task_paths[key]):
                if self.camera in root_path:  # for input and depth
                    if "objdet" in self.task_paths[key]:
                        self.paths_dict[key].extend(
                            sorted(
                                [
                                    os.path.join(root_path, file)
                                    for file in files
                                    if file.split(".")[1] in ["txt"]
                                ]
                            )
                        )
                    else:
                        self.paths_dict[key].extend(
                            sorted(
                                [
                                    os.path.join(root_path, file)
                                    for file in files
                                    if file.split(".")[1] in ["png", "jpg"]
                                ]
                            )
                        )
            self.paths_dict[key] = sorted(self.paths_dict[key])

    def _filter_data_paths(self):
        task_root_paths = {
            task: os.path.abspath(self.task_paths[task])
            for task in self.task_paths.keys()
        }
        task_extension = {
            task: self.paths_dict[task][0][-4:] for task in self.task_paths.keys()
        }
        task_unique_data = {task: set() for task in self.task_paths.keys()}
        for task, paths in self.paths_dict.items():
            for path in paths:
                task_unique_data[task].add(path.replace(task_root_paths[task], "")[:-4])

        final_paths = task_unique_data["input"]
        for task, unique_data in task_unique_data.items():
            final_paths = final_paths.intersection(unique_data)

        for task in task_root_paths.keys():
            self.paths_dict[task] = sorted(
                [
                    task_root_paths[task] + path + task_extension[task]
                    for path in final_paths
                ]
            )

        self.length = len(final_paths)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx):
        load_functions = KITTIutils.load_utils(list(self.paths_dict.keys()))
        task_item = dict.fromkeys(self.paths_dict.keys())
        for task in self.paths_dict.keys():
            task_item[task] = self.task_transform[task](
                load_functions[task](self.paths_dict[task][idx])
            )

        return task_item


if __name__ == "__main__":
    kitti_dataset = KittiDataset(
        {
            "input": "./data/kitti/input",  # ../../datasets/kitti_data/
            "depth": "./data/kitti/depth/train",
            # "objdet": "./data/kitti/objdet/train",
        },
        {
            "input": [
                # "ToTensor",
                "Crop",
            ],
            "depth": [
                # "ToTensor",
                "Crop",
            ],
            # "objdet": [
            #     "ToTensor",
            # ],
        },
        "image_02",
    )

    train_dataloader = DataLoader(
        dataset=kitti_dataset, batch_size=1, shuffle=True, num_workers=1
    )
    sample = next(iter(train_dataloader))
    breakpoint()
