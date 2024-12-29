import os
import shutil
import argparse
from pathlib import Path
import numpy as np


def map_paths(mapping_path, data_path, objdet_label_path, split_ratio):
    mapping_path = os.path.abspath(Path(mapping_path))
    input_root = os.path.abspath(Path(data_path) / "input")

    with open(f"{mapping_path}/train_mapping.txt") as file:
        lines = file.readlines()

    dataset_length = len(lines)
    indices = np.random.permutation(
        np.arange(0, dataset_length)
    )  # TODO: maybe better split?
    train_length = dataset_length * split_ratio // 100

    for line_idx in range(train_length):
        train_idx = indices[line_idx]
        txt_filename = (6 - len(str(train_idx))) * "0" + str(train_idx) + ".txt"
        source = os.path.join(os.path.abspath(objdet_label_path), txt_filename)
        line_split = lines[train_idx].strip().split()
        corresponding_raw_folder = "/".join(line_split[:-1])
        corresponding_frame = line_split[-1] + ".txt"
        target = os.path.join(
            os.path.abspath(data_path),
            "objdet",
            "train",
            corresponding_raw_folder,
            "image_02",
            "data",
        )
        if not os.path.isdir(target):
            os.makedirs(target)
        shutil.move(source, target)
        os.rename(f"{target}/{txt_filename}", f"{target}/{corresponding_frame}")

    for line_idx in range(train_length, dataset_length):
        val_idx = indices[line_idx]
        txt_filename = (6 - len(str(val_idx))) * "0" + str(val_idx) + ".txt"
        source = os.path.join(os.path.abspath(objdet_label_path), txt_filename)
        line_split = lines[val_idx].strip().split()
        corresponding_raw_folder = "/".join(line_split[:-1])
        corresponding_frame = line_split[-1] + ".txt"
        target = os.path.join(
            os.path.abspath(data_path),
            "objdet",
            "val",
            corresponding_raw_folder,
            "image_02",
            "data",
        )
        if not os.path.isdir(target):
            os.makedirs(target)
        shutil.move(source, target)
        os.rename(f"{target}/{txt_filename}", f"{target}/{corresponding_frame}")
    shutil.rmtree(
        os.path.abspath("/".join(objdet_label_path.split("/")[:-1]))
    )  # ("training/label_2"), we want to delete training/ and the subdirectory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Object Detection Structure formatter",
        description="structures the data folder for object detection task",
    )
    # unzip data_object_label_2.zip to data/kitti folder
    # unzip train_mapping.txt to data/kitti folder
    # if ran from master-thesis file
    parser.add_argument(
        "--mapping",
        type=str,
        default="./data/kitti",
        help="Path to train_mapping.txt",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data/kitti",
        help="Path to the KITTI data folder",
    )
    parser.add_argument(
        "--label-path",
        type=str,
        default="./data/kitti/training/label_2",
        help="Path to object detection labels",
    )
    parser.add_argument(
        "--split-ratio",
        type=int,
        default=80,
        help="train/val split ratio",
    )

    args = parser.parse_args()
    map_paths(args.mapping, args.data_path, args.label_path, args.split_ratio)
