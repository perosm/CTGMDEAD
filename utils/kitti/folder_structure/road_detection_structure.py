import shutil
import argparse
import pathlib


def main():
    args = _parse_args()

    path_informations = []
    for file_txt in args.devkit_road_mapping.absolute().iterdir():
        with open(file_txt, "r") as file:
            path_informations.extend(file.readlines())

    gt_images = [
        gt_image
        for gt_image in sorted(args.data_road_gt.iterdir())
        if gt_image.is_file()
    ]
    root_dir_name = pathlib.Path("./data/kitti/road_detection")
    for path_info, gt_image in zip(path_informations, gt_images):
        date_drive_num, image_number = path_info.strip().split(" ")
        date_drive_num_parts = date_drive_num.split("_")
        date = pathlib.Path("_".join(date_drive_num_parts[0:3]))
        date_drive_num = pathlib.Path(f"{date_drive_num}_sync")
        full_source_path_name = pathlib.Path(gt_image).absolute()
        full_destination_path_name = (
            root_dir_name / date / date_drive_num / "camera_02" / "data"
        ).absolute()
        full_destination_path_name.mkdir(parents=True, exist_ok=True)
        image_name = f"{image_number}.png"
        full_source_path_name.rename(full_destination_path_name / image_name)

    shutil.rmtree(args.data_road_gt.parent.parent)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-drg",
        "--data-road-gt",
        type=pathlib.Path,
        default=pathlib.Path(
            "./data/kitti/data_road/training/gt_image_2"
        ),  # left camera image
        help="Path to road segmentation ground truth.",
    )
    parser.add_argument(
        "-drm",
        "--devkit-road-mapping",
        type=pathlib.Path,
        default=pathlib.Path("./data/kitti/devkit_road_mapping/training/"),
        help="Path to mapping files for ground truth.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
