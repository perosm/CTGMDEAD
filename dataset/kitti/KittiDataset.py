"""
Depth maps (annotated and raw Velodyne scans) are saved as uint16 PNG images.
A 0 value indicates an invalid pixel
(ie, no ground truth exists, or the estimation algorithm didn't produce an
estimate for that pixel). Otherwise, the depth for a pixel can be computed
in meters by converting the uint16 value to float and dividing it by 256.0:
disp(u,v)  = ((float)I(u,v))/256.0;
valid(u,v) = I(u,v)>0;


If you unzip all downloaded files from the KITTI vision benchmark website
into the same base directory, your folder structure will look like this:

|-- devkit
|-- test_depth_completion_anonymous
  |-- image
    |-- 0000000000.png
    |-- ...
    |-- 0000000999.png
  |-- velodyne_raw
    |-- 0000000000.png
    |-- ...
    |-- 0000000999.png
|-- test_depth_prediction_anonymous
  |-- image
    |-- 0000000000.png
    |-- ...
    |-- 0000000999.png
|-- train
  |-- 2011_xx_xx_drive_xxxx_sync
    |-- proj_depth
      |-- groundtruth           # "groundtruth" describes our annotated depth maps
        |-- image_02            # image_02 is the depth map for the left camera
          |-- 0000000005.png    # image IDs start at 5 because we accumulate 11 frames
          |-- ...               # .. which is +-5 around the current frame ;)
        |-- image_03            # image_02 is the depth map for the right camera
          |-- 0000000005.png
          |-- ...
      |-- velodyne_raw          # this contains projected and temporally unrolled
        |-- image_02            # raw Velodyne laser scans
          |-- 0000000005.png
          |-- ...
        |-- image_03
          |-- 0000000005.png
          |-- ...
  |-- ... (all drives of all days in the raw KITTI dataset)
|-- val
  |-- (same as in train)
|-- val_selection_cropped       # 1000 images of size 1216x352, cropped and manually
  |-- groundtruth_depth         # selected frames from from the full validation split
    |-- 2011_xx_xx_drive_xxxx_sync_groundtruth_depth_xxxxxxxxxx_image_0x.png
    |-- ...
  |-- image
    |-- 2011_xx_xx_drive_xxxx_sync_groundtruth_depth_xxxxxxxxxx_image_0x.png
    |-- ...
  |-- velodyne_raw
    |-- 2011_xx_xx_drive_xxxx_sync_groundtruth_depth_xxxxxxxxxx_image_0x.png
    |-- ...

"""

import os
import torch
import sys
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from typing import Dict, List


class KittiDataset(Dataset):
    def __init__(
        self,
        # transforms: Dict[str, list],
        paths: Dict[str, str],
        camera: str = "03",
    ) -> None:
        """
        Args:
          - transforms: dictionary of transforms for each of the network tasks.
          - camera: string which represents left camera ('02') or right camera ('03')
          - paths: paths to input data and ground truth for neural networks tasks (e.g. object detection, depth estimation...)
                   paired with the path to the folder where the ground truth for each of the tasks is stored.

        Returns:
          - None
        """
        # self.transforms = transforms
        self.paths = paths
        self.camera = camera

        self._load_data_paths()

    def _load_data_paths(self):
        length = 0
        paths_dict = {key: [] for key in self.paths.keys()}
        for key in self.paths:
            for root_path, dirs, files in os.walk(self.paths[key]):
                if self.camera in root_path:  # for input and depth
                    paths_dict[key].extend(
                        sorted([os.path.join(root_path, file) for file in files])
                    )
                    length += len(files)
            paths_dict[key] = sorted(paths_dict[key])

        for i in range(0, len(paths_dict["input"]), 5):
            print(paths_dict["input"][i])
            print(paths_dict["depth"][i])
            breakpoint()

        print(len(paths_dict["input"]), len(paths_dict["depth"]))
        self.length = length

    def __len__(self):
        return len(self.length)

    def __getitem__(self, index):
        pass


if __name__ == "__main__":
    kitti_dataset = KittiDataset(
        {
            "input": "./data/kitti/input",
            "depth": "./data/kitti/depth/train",
        },
        "image_03",
    )
