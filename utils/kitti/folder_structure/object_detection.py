import shutil
import argparse
import pathlib
import numpy as np
import cv2
import matplotlib.pyplot as plt


def main():
    args = _parse_args()

    input_images_dir = args.input_images_dir
    mapping_path = args.devkit_object_detection_mapping
    ground_truth_dir = args.label_object_detection
    calibration_dir = args.object_detection_calibration
    destination_root_folder = args.destination_root_folder

    with open(mapping_path / "train_mapping.txt", "r") as file:
        mapping_info = [line.strip().split(" ") for line in file.readlines()]

    with open(mapping_path / "train_rand.txt", "r") as file:
        mapping_info_indexes = [
            int(number) for number in file.readline().strip().split(",")
        ]
    ground_truth_paths = sorted([file for file in ground_truth_dir.iterdir()])
    calibration_paths = sorted([file for file in calibration_dir.iterdir()])

    for (
        frame_index,
        ground_truth_path,
        calibration_path,
    ) in zip(mapping_info_indexes, ground_truth_paths, calibration_paths):
        date, drive, frame = mapping_info[frame_index - 1]
        subfolders = pathlib.Path(f"{date}/{drive}/image_02/data")
        ground_truth_destination_dir = (
            destination_root_folder / "ground_truth" / subfolders
        )
        ground_truth_destination_dir.mkdir(parents=True, exist_ok=True)
        calibration_destination_dir = (
            destination_root_folder / "calibrations" / subfolders
        )
        calibration_destination_dir.mkdir(parents=True, exist_ok=True)
        ground_truth_path.replace(
            ground_truth_destination_dir / f"{frame}{ground_truth_path.suffix}"
        )
        calibration_path.replace(
            calibration_destination_dir / f"{frame}{calibration_path.suffix}"
        )

    shutil.rmtree(mapping_path.parent)
    shutil.rmtree(ground_truth_dir.parent.parent)
    shutil.rmtree(calibration_dir.parent.parent)
    num_files = sum(1 for _ in destination_root_folder.rglob("*.txt") if _.is_file())

    print(f"Number of files: {num_files / 2}")
    print(f"Number of mappings: {len(mapping_info_indexes)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-iid",
        "--input-images-dir",
        type=pathlib.Path,
        default=pathlib.Path("./data/kitti/input"),  # left camera images
        help="Path to input images root folder.",
    )
    parser.add_argument(
        "-lod",
        "--label-object-detection",
        type=pathlib.Path,
        default=pathlib.Path(
            "./data/kitti/data_object_label_2/training/label_2"
        ),  # left camera images
        help="Path to object detection ground truth.",
    )
    parser.add_argument(
        "-odc",
        "--object-detection-calibration",
        type=pathlib.Path,
        default=pathlib.Path(
            "./data/kitti/data_object_calib/training/calib"
        ),  # left camera images
        help="Path to object detection calibration files.",
    )
    parser.add_argument(
        "-drf",
        "--destination-root-folder",
        type=pathlib.Path,
        default=pathlib.Path("./data/kitti/object_detection"),
        help="Destination root folder for all the corresponding files.",
    )
    parser.add_argument(
        "-dodm",
        "--devkit-object-detection-mapping",
        type=pathlib.Path,
        default=pathlib.Path("./data/kitti/devkit_object/mapping"),
        help="Path to mapping folder. Contains information how to map labels to raw data.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
