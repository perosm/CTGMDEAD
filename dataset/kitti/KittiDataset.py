import os
import pathlib
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import dataset.dataset_utils as KITTIutils
import sys


class KittiDataset(Dataset):

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
        self.camera = camera

        self._load_data_paths()
        self._filter_data_paths()
        self.load_functions = KITTIutils.load_utils(list(self.paths_dict.keys()))

    def _configure_transforms(
        self, task_transform: dict
    ) -> dict[str, transforms.Compose]:
        return {
            task: transforms.Compose(*[task_transform[task]]) for task in task_transform
        }

    def _load_data_paths(self) -> None:
        self.paths_dict = {key: [] for key in self.task_paths.keys()}
        for task in self.task_paths.keys():
            for root_path, dirs, files in self.task_paths[task].walk():
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
                task_unique_data[task].add("/".join(path.with_suffix("").parts[-5:]))

        final_paths = task_unique_data["input"]
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
        return self.length

    def __getitem__(self, idx):
        task_item = dict.fromkeys(self.paths_dict.keys())
        for task in self.paths_dict.keys():
            task_item[task] = self.task_transform[task](
                self.load_functions[task](self.paths_dict[task][idx])
            )
            print(self.paths_dict[task][idx])
        # task_item.update({"path": self.paths_dict[idx]})
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
