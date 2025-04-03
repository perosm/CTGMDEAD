import shutil
import argparse
import pathlib


def main():
    args = _parse_args()

    destination_root_dir_name = pathlib.Path("./data/kitti/object_detection_3d")

    with open(args.devkit_object_detection_3d_mapping) as txt_file:
        lines = txt_file.readlines()

    ground_truth_files = sorted(
        [file for file in args.data_object_detection_3d.iterdir()]
    )

    for line, ground_truth in zip(lines, ground_truth_files):
        date, date_drive_number, image_number = line.strip().split(" ")
        full_source_path_name = pathlib.Path(ground_truth).absolute()
        full_destination_path_name = (
            destination_root_dir_name / date / date_drive_number / "image_02" / "data"
        ).absolute()
        full_destination_path_name.mkdir(parents=True, exist_ok=True)
        image_name = f"{image_number}.txt"
        full_source_path_name.rename(full_destination_path_name / image_name)

    shutil.rmtree(args.data_object_detection_3d.parent.parent)
    shutil.rmtree(args.devkit_object_detection_3d_mapping.parent.parent)
    num_files = sum(1 for _ in destination_root_dir_name.rglob("*") if _.is_file())

    print(f"Number of files: {num_files}")
    print(f"Number of mappings: {len(lines)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dod",
        "--data-object-detection-3d",
        type=pathlib.Path,
        default=pathlib.Path(
            "./data/kitti/data_object_label_2/training/label_2"
        ),  # left camera images
        help="Path to 3d object detection ground truth.",
    )
    parser.add_argument(
        "-dodm",
        "--devkit-object-detection-3d-mapping",
        type=pathlib.Path,
        default=pathlib.Path("./data/kitti/devkit_object/mapping/train_mapping.txt"),
        help="Path to information about mapping ground truth to images.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
