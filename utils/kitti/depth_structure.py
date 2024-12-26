import os
import shutil

# root folder is master-thesis
root_folder = os.path.join(os.getcwd(), "data", "kitti")  # /master-thesis/data/kitti
root_depth_train = os.path.join(root_folder, "train")  # /master-thesis/data/kitti/train
root_depth_val = os.path.join(root_folder, "val")  # /master-thesis/data/kitti/val
new_root_depth = os.path.join(root_folder, "depth")  # /master-thesis/data/kitti/depth

"""
moving image_02 & image_03 folders
from /train/20xx_xx_xx_drive_xxxx_sync/proj_depth/groundtruth
to   /train/20xx_xx_xx_drive_xxxx_sync
"""
for root_depth_tv in [root_depth_train, root_depth_val]:
    if os.path.exists(root_depth_tv):
        for subdir in sorted(os.listdir(root_depth_tv)):
            drive_dir = os.path.join(root_depth_tv, subdir)
            groundtruth_dir = os.path.join(drive_dir, "proj_depth", "groundtruth")
            if os.path.isdir(groundtruth_dir):
                for folder in ["image_02", "image_03"]:
                    source_path = os.path.join(groundtruth_dir, folder)
                    target_path = os.path.join(drive_dir, folder)
                    if not os.path.exists(target_path):
                        os.makedirs(target_path)
                        shutil.move(source_path, target_path)

drives = [
    "2011_09_26",
    "2011_09_28",
    "2011_09_29",
    "2011_09_30",
    "2011_10_03",
]

"""
moving data
from /train/20xx_xx_xx_drive_xxxx_sync/
to   /depth/train/20xx_xx_xx/20xx_xx_xx_drive_xxxx_sync
"""
for drive in sorted(drives):
    for folder in ["train", "val"]:
        drive_dir = os.path.join(new_root_depth, folder, drive)
        if not os.path.isdir(drive_dir):
            os.makedirs(drive_dir)
        source_folders = os.path.join(root_folder, folder)
        if os.path.exists(source_folders):
            for sf in sorted(os.listdir(source_folders)):
                if drive in sf:
                    shutil.move(os.path.join(source_folders, sf), drive_dir)

if os.path.exists(root_depth_train):
    os.rmdir(root_depth_train)
if os.path.exists(root_depth_val):
    os.rmdir(root_depth_val)

for root, dirs, files in os.walk(new_root_depth):
    for dir in dirs:
        if not os.listdir(os.path.join(root, dir)):
            os.rmdir(os.path.join(root, dir))
