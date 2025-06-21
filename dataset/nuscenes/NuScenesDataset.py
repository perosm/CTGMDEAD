import os

import torch
import pathlib
from torch.utils.data.dataset import Dataset
from dataset.nuscenes.nuscenes_devkit.nuscenes.nuscenes import NuScenes
from dataset.nuscenes.nuscenes_devkit.nuimages.nuimages import NuImages

from utils.shared.enums import TaskEnum


class NuScenesNuImagesDataset(Dataset):

    def __init__(
        self,
        tasks: list[str] = [
            TaskEnum.input,
            TaskEnum.depth,
            TaskEnum.object_detection,
        ],  # road detection
        version: str = "v1.0-mini",
        nuscenes_kitti_dataroot: pathlib.Path = "./data/nuscenes/nuscenes_kitti",
        nuimages_dataroot: pathlib.Path = "./data/nuscenes/nuimages",
    ):
        super().__init__()
        self.nuscenes_kitti_root = nuscenes_kitti_dataroot
        self.nuimages = NuImages(
            version=version, dataroot=nuimages_dataroot, verbose=True, lazy=True
        )  # TODO: change verbose=False

        self.nuscenes_sample_list = self._read_nuscenes_sample_list()
        self.nuimages_sample_list = self._read_nuimages_sample_list()
        self.sample_list = self.nuscenes_sample_list + self.nuimages_sample_list

    def _read_nuscenes_sample_list(self) -> list[str]:
        """
        Sample list for nuscenes is shared for:
            - input
            - depth
            - object_detection

        Returns:
            NuScenes sample list.
        """
        sample_list = sorted(os.listdir(f"{self.nuscenes_kitti_root}/image_2"))

        return sample_list

    def _read_nuimages_sample_list(self) -> list[str]:
        sample_list = []
        for sample_record in self.nuimages.sample:
            sample_list.append(sample_record["token"])

        return sample_list

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        return super().__getitem__(index)


if __name__ == "__main__":
    pass
