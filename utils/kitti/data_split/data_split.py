"""
Let KITTI dataset be split into three categories:
 _____________________________________________________
|                      |            |                 |
|          A1          |     A2     |       A3        |
|                      |            |                 |
|_____________________________________________________|

Where A1 will be used to train the autoencoder for the depth estimation task,
A3 will be used to additionally train the road detection decoder, 2D and 3D
detection heads. Every network task will be evaluated on A2.

In order for the script to work, all directories need to be in the same format:

└── date
    ├── date_drive
    │ └── image_02
    │     └── data

e.g.
└── 2011_10_03
    ├── 2011_10_03_drive_0027_sync
    │ └── image_02
    │     └── data
    ├── 2011_10_03_drive_0034_sync
    │ └── image_02
    │     └── data
    ├── 2011_10_03_drive_0042_sync
    │ └── image_02
    │     └── data
    └── 2011_10_03_drive_0047_sync
        └── image_02
            └── data
"""

import pathlib
import argparse
import random
import json
from collections import defaultdict

SEED = 42
CAMERA = "image_02"
KITTI_CATEGORIES_PATH = "./utils/kitti/data_split/kitti_categories.json"

DATE_INDEX = -4
DRIVE_INDEX = -3
DRIVE_NUMBER_INDEX = -2
UNIQUE_IDENTIFIER_SLICE = slice(-5, None)


def main() -> None:
    args = _parse_args()
    a1_ratio, a2_ratio, a3_ratio = args.a1_ratio, args.a2_ratio, args.a3_ratio
    assert a1_ratio + a2_ratio == 1.0, "a1_ratio + a2_ratio need to equal to one!"
    assert a2_ratio + a3_ratio == 1.0, "a2_ratio + a3_ratio need to equal to one!"
    assert a1_ratio == a3_ratio, "a1_ratio needs to be equal to a3_ratio!"

    split_dataset(
        a1_ratio=a1_ratio,
        a2_ratio=a2_ratio,
        a3_ratio=a3_ratio,
        depth_root_dir=args.depth_root_dir,
        object_detection_root_dir=args.object_detection_root_dir,
        semantic_segmentation_root_dir=args.semantic_segmentation_root_dir,
        save_path=args.save_path,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-a1r", "--a1-ratio", default=0.9, type=float)
    parser.add_argument("-a2r", "--a2-ratio", default=0.1, type=float)
    parser.add_argument("-a3r", "--a3-ratio", default=0.9, type=float)
    parser.add_argument(
        "--depth-root-dir", default="./data/kitti/depth", type=pathlib.Path
    )
    parser.add_argument(
        "--object-detection-root-dir",
        default="./data/kitti/object_detection/ground_truth",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--semantic-segmentation-root-dir",
        default="./data/kitti/semantic_segmentation",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--save-path",
        default="./data/kitti/sample_lists",
        type=pathlib.Path,
    )
    return parser.parse_args()


def split_dataset(
    a1_ratio: float,
    a2_ratio: float,
    a3_ratio: float,
    depth_root_dir: pathlib.Path,
    object_detection_root_dir: pathlib.Path,
    semantic_segmentation_root_dir: pathlib.Path,
    save_path: pathlib.Path,
):
    random.seed(SEED)
    with open(KITTI_CATEGORIES_PATH, "r") as file:
        kitti_categories = json.load(file)

    # Semantic segmentation data
    semantic_segmentation_date_sample_list = _read_data(
        root_dir=semantic_segmentation_root_dir, extension=".png"
    )
    permuted_semantic_segmentation_date_drive_sample_list = {
        date_drive_number: random.sample(sorted(data_paths), len(data_paths))
        for date_drive_number, data_paths in semantic_segmentation_date_sample_list.items()
    }
    semseg_date_drive_sample_list_train, semseg_date_drive_sample_list_val = (
        _split_dataset(
            ratio1=a3_ratio,
            ratio2=a2_ratio,
            date_drive_sample_list=permuted_semantic_segmentation_date_drive_sample_list,
        )
    )
    # Object detection data
    object_detection_date_drive_sample_list = _read_data(
        root_dir=object_detection_root_dir, extension=".txt"
    )
    permuted_object_detection_date_drive_sample_list = {
        date_drive_number: random.sample(sorted(data_paths), len(data_paths))
        for date_drive_number, data_paths in object_detection_date_drive_sample_list.items()
    }
    objdet_date_drive_sample_list_train, objdet_date_drive_sample_list_val = (
        _split_dataset(
            ratio1=a3_ratio,
            ratio2=a2_ratio,
            date_drive_sample_list=permuted_object_detection_date_drive_sample_list,
        )
    )
    # Depth data
    depth_date_drive_sample_list = _read_data(root_dir=depth_root_dir, extension=".png")
    permuted_depth_date_drive_sample_list = {
        date_drive_number: random.sample(sorted(data_paths), len(data_paths))
        for date_drive_number, data_paths in depth_date_drive_sample_list.items()
    }

    depth_date_drive_sample_list_train, depth_date_drive_sample_list_val = (
        _split_dataset(
            ratio1=a1_ratio,
            ratio2=a2_ratio,
            date_drive_sample_list=permuted_depth_date_drive_sample_list,
        )
    )
    # If samples are overlapping between depth train and objdet/semseg train
    # we remove them from depth train and keep them inside objdet/semseg train
    # 1) Semantic segmentation and depth
    (
        depth_date_drive_sample_list_train,
        semseg_date_drive_sample_list_train,
        date_drive_overlapping_samples,
    ) = _remove_ovelapping_samples(
        date_drive_sample_list_train=depth_date_drive_sample_list_train,
        date_drive_sample_list_val=semseg_date_drive_sample_list_train,
    )

    semseg_date_drive_sample_list_train = _add_overlapping_samples(
        date_drive_sample_list=semseg_date_drive_sample_list_train,
        date_drive_overlapping_samples=date_drive_overlapping_samples,
    )

    # 2) Object detection and depth
    (
        depth_date_drive_sample_list_train,
        objdet_date_drive_sample_list_train,
        date_drive_overlapping_samples,
    ) = _remove_ovelapping_samples(
        date_drive_sample_list_train=depth_date_drive_sample_list_train,
        date_drive_sample_list_val=objdet_date_drive_sample_list_train,
    )

    objdet_date_drive_sample_list_train = _add_overlapping_samples(
        date_drive_sample_list=objdet_date_drive_sample_list_train,
        date_drive_overlapping_samples=date_drive_overlapping_samples,
    )
    # If samples between are overlapping we move them to from train to validation
    # 1) depth train and semantic segmentation validation
    (
        depth_date_drive_sample_list_train,
        semseg_date_drive_sample_list_val,
        date_drive_overlapping_samples,
    ) = _remove_ovelapping_samples(
        date_drive_sample_list_train=depth_date_drive_sample_list_train,
        date_drive_sample_list_val=semseg_date_drive_sample_list_val,
    )
    semseg_date_drive_sample_list_val = _add_overlapping_samples(
        date_drive_sample_list=semseg_date_drive_sample_list_val,
        date_drive_overlapping_samples=date_drive_overlapping_samples,
    )
    # 2) depth train and object detection validation
    (
        depth_date_drive_sample_list_train,
        objdet_date_drive_sample_list_val,
        date_drive_overlapping_samples,
    ) = _remove_ovelapping_samples(
        date_drive_sample_list_train=depth_date_drive_sample_list_train,
        date_drive_sample_list_val=objdet_date_drive_sample_list_val,
    )
    objdet_date_drive_sample_list_val = _add_overlapping_samples(
        date_drive_sample_list=objdet_date_drive_sample_list_val,
        date_drive_overlapping_samples=date_drive_overlapping_samples,
    )
    # If samples between are overlapping we move them from validation to train
    # due to low number of samples in object detection and semantic segmentation datasets
    # 3) object detection train and semantic segmentation validation
    (
        objdet_date_drive_sample_list_train,
        semseg_date_drive_sample_list_val,
        date_drive_overlapping_samples,
    ) = _remove_ovelapping_samples(
        date_drive_sample_list_train=objdet_date_drive_sample_list_train,
        date_drive_sample_list_val=semseg_date_drive_sample_list_val,
    )
    objdet_date_drive_sample_list_train = _add_overlapping_samples(
        date_drive_sample_list=objdet_date_drive_sample_list_train,
        date_drive_overlapping_samples=date_drive_overlapping_samples,
    )
    # 4) object detection train and depth validation and object detection
    (
        objdet_date_drive_sample_list_train,
        depth_date_drive_sample_list_val,
        date_drive_overlapping_samples,
    ) = _remove_ovelapping_samples(
        date_drive_sample_list_train=objdet_date_drive_sample_list_train,
        date_drive_sample_list_val=depth_date_drive_sample_list_val,
    )
    objdet_date_drive_sample_list_train = _add_overlapping_samples(
        date_drive_sample_list=objdet_date_drive_sample_list_train,
        date_drive_overlapping_samples=date_drive_overlapping_samples,
    )
    # 5) semantic segmentation train and depth validation
    (
        semseg_date_drive_sample_list_train,
        depth_date_drive_sample_list_val,
        date_drive_overlapping_samples,
    ) = _remove_ovelapping_samples(
        date_drive_sample_list_train=semseg_date_drive_sample_list_train,
        date_drive_sample_list_val=depth_date_drive_sample_list_val,
    )
    semseg_date_drive_sample_list_train = _add_overlapping_samples(
        date_drive_sample_list=semseg_date_drive_sample_list_train,
        date_drive_overlapping_samples=date_drive_overlapping_samples,
    )
    # Data per task statistics
    _get_sample_list_statistics(
        date_drive_sample_list_train=depth_date_drive_sample_list_train,
        date_drive_sample_list_val=depth_date_drive_sample_list_val,
        kitti_categories=kitti_categories,
        task="Depth estimation",
    )
    _get_sample_list_statistics(
        date_drive_sample_list_train=semseg_date_drive_sample_list_train,
        date_drive_sample_list_val=semseg_date_drive_sample_list_val,
        kitti_categories=kitti_categories,
        task="Semantic segmentation",
    )
    _get_sample_list_statistics(
        date_drive_sample_list_train=objdet_date_drive_sample_list_train,
        date_drive_sample_list_val=objdet_date_drive_sample_list_val,
        kitti_categories=kitti_categories,
        task="Object detection",
    )

    # Flatten dicts of {depth/semantic_segmentation/object_detection}_date_drive_sample_list_{train/val}
    # into list {depth/semantic_segmentation/object_detection}_sample_list_{train/val} and save them as a .txt file
    depth_sample_list_train = _flatten_into_list(
        date_drive_sample_list=depth_date_drive_sample_list_train
    )
    depth_sample_list_val = _flatten_into_list(
        date_drive_sample_list=depth_date_drive_sample_list_val
    )
    semseg_sample_list_train = _flatten_into_list(
        date_drive_sample_list=semseg_date_drive_sample_list_train
    )
    semseg_sample_list_val = _flatten_into_list(
        date_drive_sample_list=semseg_date_drive_sample_list_val
    )
    objdet_sample_list_train = _flatten_into_list(
        date_drive_sample_list=objdet_date_drive_sample_list_train
    )
    objdet_sample_list_val = _flatten_into_list(
        date_drive_sample_list=objdet_date_drive_sample_list_val
    )

    assert (
        set(depth_sample_list_train).intersection(set(semseg_sample_list_train))
        == set()
    ), f"Intersection between depth_sample_list_train and semseg_sample_list_train should be empty!"
    assert (
        set(depth_sample_list_train).intersection(set(objdet_sample_list_train))
        == set()
    ), f"Intersection between depth_sample_list_train and objdet_sample_list_train should be empty!"
    assert (
        set(depth_sample_list_train).intersection(set(depth_sample_list_val)) == set()
    ), f"Intersection between depth_sample_list_train and depth_sample_list_val should be empty!"
    assert (
        set(depth_sample_list_train).intersection(
            set(semseg_date_drive_sample_list_val)
        )
        == set()
    ), f"Intersection between depth_sample_list_train and semseg_sample_list_val should be empty!"
    assert (
        set(depth_sample_list_train).intersection(set(objdet_sample_list_val)) == set()
    ), f"Intersection between depth_sample_list_train and objdet_sample_list_val should be empty!"
    assert (
        set(semseg_sample_list_train).intersection(set(semseg_sample_list_val)) == set()
    ), f"Intersection between semseg_sample_list_train and semseg_sample_list_val should be empty!"
    assert (
        set(semseg_sample_list_train).intersection(set(objdet_sample_list_val)) == set()
    ), f"Intersection between semseg_sample_list_train and objdet_sample_list_val should be empty!"
    assert (
        set(objdet_sample_list_train).intersection(set(objdet_sample_list_val)) == set()
    ), f"Intersection between objdet_sample_list_train and objdet_sample_list_val should be empty!"

    _save_data(
        sample_list=depth_sample_list_train,
        save_path=save_path / "train",
        filename="depth_sample_list.txt",
    )
    _save_data(
        sample_list=depth_sample_list_val,
        save_path=save_path / "val",
        filename="depth_sample_list.txt",
    )
    _save_data(
        sample_list=semseg_sample_list_train,
        save_path=save_path / "train",
        filename="semseg_sample_list.txt",
    )
    _save_data(
        sample_list=semseg_sample_list_val,
        save_path=save_path / "val",
        filename="semseg_sample_list.txt",
    )
    _save_data(
        sample_list=objdet_sample_list_train,
        save_path=save_path / "train",
        filename="objdet_sample_list.txt",
    )
    _save_data(
        sample_list=objdet_sample_list_val,
        save_path=save_path / "val",
        filename="objdet_sample_list.txt",
    )


def _read_data(root_dir: pathlib.Path, extension: str) -> dict[str, list[str]]:
    date_drive_sample_list = defaultdict(list)

    for dirpath, _, filenames in root_dir.walk():
        for filename in filenames:
            if filename.endswith(extension):
                date = dirpath.parts[DATE_INDEX]
                drive = dirpath.parts[DRIVE_INDEX]
                drive_number = drive.split("_")[DRIVE_NUMBER_INDEX]
                path = dirpath / filename
                path = _remove_extension(path)
                path = "/".join(path.parts[UNIQUE_IDENTIFIER_SLICE])
                date_drive_sample_list[f"{date}_{drive_number}"].append(path)

    return date_drive_sample_list


def _remove_extension(path: pathlib.Path) -> pathlib.Path:
    file_without_extension = path.stem
    return path.parent / file_without_extension


def _split_dataset(
    ratio1: float,
    ratio2: float,
    date_drive_sample_list: dict[str, list[str]],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    assert ratio1 + ratio2 == 1, "ratio1 + ratio2 need to be equal to 1!"
    date_drive_sample_list1 = defaultdict(list)
    date_drive_sample_list2 = defaultdict(list)

    for date_drive, sample_list in date_drive_sample_list.items():
        num_samples = len(sample_list)
        num_samples1 = int(ratio1 * num_samples)
        date_drive_sample_list1[date_drive] = sample_list[:num_samples1]
        date_drive_sample_list2[date_drive] = sample_list[num_samples1:]

    return date_drive_sample_list1, date_drive_sample_list2


def _remove_ovelapping_samples(
    date_drive_sample_list_train: dict[str, list[str]],
    date_drive_sample_list_val: dict[str, list[str]],
) -> dict[str, list[str]]:
    date_drive_overlapping_samples = defaultdict(list)
    date_drives_train, date_drives_val = (
        set(date_drive_sample_list_train.keys()),
        set(date_drive_sample_list_val.keys()),
    )
    date_drives = date_drives_train.intersection(date_drives_val)

    for date_drive in date_drives:
        sample_list_train = set(date_drive_sample_list_train[date_drive])
        sample_list_val = set(date_drive_sample_list_val[date_drive])
        date_drive_overlapping_samples[date_drive] = sample_list_train.intersection(
            sample_list_val
        )

    for date_drive in date_drives:
        for overlapping_sample in date_drive_overlapping_samples[date_drive]:
            date_drive_sample_list_train[date_drive].remove(overlapping_sample)
            date_drive_sample_list_val[date_drive].remove(overlapping_sample)

    return (
        date_drive_sample_list_train,
        date_drive_sample_list_val,
        date_drive_overlapping_samples,
    )


def _add_overlapping_samples(
    date_drive_sample_list: dict[str, list[str]],
    date_drive_overlapping_samples: dict[str, list[str]],
) -> dict[str, list[str]]:
    for date_drive in date_drive_overlapping_samples:
        date_drive_sample_list[date_drive].extend(
            date_drive_overlapping_samples[date_drive]
        )

    return date_drive_sample_list


def _get_sample_list_statistics(
    date_drive_sample_list_train: dict[str, list[str]],
    date_drive_sample_list_val: dict[str, list[str]],
    kitti_categories: dict[str, list],
    task: str,
):

    print(task)
    for kitti_category, date_drives in kitti_categories.items():
        train_samples_per_category = 0
        val_samples_per_category = 0
        total_samples_per_category = 0
        for date_drive in date_drives:
            train_samples_per_category += len(date_drive_sample_list_train[date_drive])
            val_samples_per_category += len(date_drive_sample_list_val[date_drive])

        total_samples_per_category += (
            train_samples_per_category + val_samples_per_category
        )
        print(f"\t{kitti_category}:")
        print(
            f"\t\tNumber of samples in train: {train_samples_per_category}/{total_samples_per_category}"
        )
        print(
            f"\t\tNumber of samples in val: {val_samples_per_category}/{total_samples_per_category}"
        )
    print("___________________________________________________________")


def _flatten_into_list(date_drive_sample_list: dict[str, list[str]]) -> list[str]:
    sample_list = []
    for date_drive in date_drive_sample_list:
        sample_list.extend(date_drive_sample_list[date_drive])

    return sorted(sample_list)


def _save_data(
    sample_list: dict[str, list[str]], save_path: pathlib.Path, filename: str
) -> None:
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / filename, "w") as file:
        file.writelines(f"{sample}\n" for sample in sample_list)


if __name__ == "__main__":
    main()
