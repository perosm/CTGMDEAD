import shutil
import argparse
import pathlib
import matplotlib.pyplot as plt

import cv2


"""
Nr.     Sequence name     Start   End
---------------------------------------
"""
INFO_DICT = {
    "00": "2011_10_03_drive_0027-000000-004540",
    "01": "2011_10_03_drive_0042-000000-001100",
    "02": "2011_10_03_drive_0034-000000-004660",
    # "03": "2011_09_26_drive_0067-000000-000800", This sequence does not exist in raw data
    "04": "2011_09_30_drive_0016-000000-000270",
    "05": "2011_09_30_drive_0018-000000-002760",
    "06": "2011_09_30_drive_0020-000000-001100",
    "07": "2011_09_30_drive_0027-000000-001100",
    "08": "2011_09_30_drive_0028-001100-005170",
    "09": "2011_09_30_drive_0033-000000-001590",
    "10": "2011_09_30_drive_0034-000000-001200",
}


def main():
    """
    Since no official mapping exists, mappings and annotations from
    https://www.zemris.fer.hr/~ssegvic/multiclod/kitti_semseg_unizg.shtml
    are used instead.
    """
    args = _parse_args()

    semantic_segmentations_label_paths = [
        file_path
        for file_path in (
            args.data_semantic_segmentation_label / "train" / "labels"
        ).iterdir()
        if file_path.is_file()
    ]

    semantic_segmentations_label_paths.extend(
        [
            file_path
            for file_path in (
                args.data_semantic_segmentation_label / "test" / "labels"
            ).iterdir()
            if file_path.is_file()
        ]
    )
    semantic_segmentations_label_paths.sort()

    input_paths = []
    for dirpath, dirname, filenames in args.input_path.walk():
        for filename in filenames:
            if filename.split(".")[-1] == "png" and "image_02" in dirpath.parts:
                input_paths.append(dirpath / filename)

    input_paths.sort()

    destination_root_folder = args.new_data_semantic_segmentation_labels
    for semseg_label_path in semantic_segmentations_label_paths:
        sequence_number, frame_number = semseg_label_path.stem.split("_")
        if sequence_number not in INFO_DICT.keys():
            continue
        sequence_name, start_frame_number, end_frame_number = INFO_DICT[
            sequence_number
        ].split("-")
        start = (
            0 if start_frame_number == "0" * 6 else int(start_frame_number.lstrip("0"))
        )
        offset = 0 if frame_number == "0" * 6 else int(frame_number.lstrip("0"))
        new_frame_number = str(start + offset)
        new_frame_name = "0" * (10 - len(new_frame_number)) + new_frame_number
        date = "_".join(sequence_name.split("_")[:3])
        drive = f"{sequence_name}_sync"
        destination_full_path = (
            destination_root_folder / date / drive / "image_02" / "data"
        ).absolute()
        destination_full_path.mkdir(parents=True, exist_ok=True)
        input_full_path = (
            args.input_path
            / date
            / drive
            / "image_02"
            / "data"
            / f"{new_frame_name}.png"
        )
        # visual_inspection(input_full_path, semseg_label_path)
        semseg_label_path = semseg_label_path.absolute()
        semseg_label_path.rename(destination_full_path / f"{new_frame_name}.png")

    shutil.rmtree(args.data_semantic_segmentation_label)
    num_files = sum(1 for _ in destination_root_folder.rglob("*") if _.is_file())
    print(f"Number of files: {num_files}")
    print(f"Number of original files: {len(semantic_segmentations_label_paths)}")


def visual_inspection(input_path: pathlib.Path, label_path: pathlib.Path) -> None:
    input_image = cv2.cvtColor(
        cv2.imread(str(input_path.absolute())), cv2.COLOR_BGR2RGB
    )
    label_image = cv2.cvtColor(
        cv2.imread(str(label_path.absolute())), cv2.COLOR_BGR2RGB
    )

    fig, ax = plt.subplots(2, 1, figsize=(10, 50))
    ax[0].imshow(input_image)
    ax[1].imshow(label_image)

    plt.show()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dsl",
        "--data-semantic-segmentation-label",
        type=pathlib.Path,
        default=pathlib.Path("./data/kitti/kitti_semseg_unizg"),
        help="Path to road segmentation labels (ground truth).",
    )
    parser.add_argument(
        "-ip",
        "--input-path",
        type=pathlib.Path,
        default=pathlib.Path("./data/kitti/input"),
        help="Path to input images.",
    )
    parser.add_argument(
        "-ndsl",
        "--new-data-semantic-segmentation-labels",
        type=pathlib.Path,
        default=pathlib.Path("./data/kitti/semantic_segmentation"),
        help="New root folder where semantic segmentation data will be stored.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
