""" """

import shutil
import pathlib
import argparse


def main():
    args = _parse_args()
    root_depth_path = args.root_depth_path
    new_root_depth_path = args.new_root_depth_path
    move_camera_02_depth_files_and_remove_unecessary(
        root_depth_path=root_depth_path, new_root_depth_path=new_root_depth_path
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-rdp",
        "--root-depth-path",
        default=pathlib.Path("./data/kitti/data_depth_annotated"),
        type=pathlib.Path,
    )
    parser.add_argument(
        "-nrdp",
        "--new-root-depth-path",
        default=pathlib.Path("./data/kitti/depth"),
        type=pathlib.Path,
    )
    return parser.parse_args()


def move_camera_02_depth_files_and_remove_unecessary(
    root_depth_path: pathlib.Path, new_root_depth_path: pathlib.Path
) -> None:
    """
    Moves 'image_02' ground truth depth directories from their original KITTI structure.
    After moving, the original root depth path is completely removed.

    This function expects the original depth files to be organized like:
    `root_depth_path/{train,val}/<date_drive_sequence>/proj_depth/groundtruth/image_02`
    (e.g., `data_depth_annotated/train/2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_02`)

    It reorganizes them into a new structure:
    `new_root_depth_path/<date_drive>/<date_drive_sequence>/image_02/data`
    (e.g., `depth/2011_09_26/2011_09_26_drive_0001_sync/image_02/data`)

    Args:
        root_depth_path: The absolute or relative path to the original root directory containing the KITTI depth annotations.
                         This directory will be deleted upon successful completion.
        new_root_depth_path: The absolute or relative path to the desired new root directory for the
                             reorganized depth data. Parent directories will be created if they don't exist.
    """

    new_root_depth_path.mkdir(parents=True, exist_ok=True)

    for dataset in ["train", "val"]:
        dataset_root_depth_path = root_depth_path / dataset
        for drive_dir in sorted(dataset_root_depth_path.iterdir()):
            drive_dir_name = drive_dir.stem
            date_dir = "_".join(drive_dir_name.split("_")[:3])
            source_path = pathlib.Path(
                drive_dir, "proj_depth", "groundtruth", "image_02"
            )
            target_path = pathlib.Path(
                new_root_depth_path,
                date_dir,
                drive_dir_name,
                "image_02",
                "data",
            )
            target_path.mkdir(parents=True, exist_ok=True)
            if source_path.exists():
                shutil.move(source_path, target_path)

    shutil.rmtree(root_depth_path)


if __name__ == "__main__":
    main()
