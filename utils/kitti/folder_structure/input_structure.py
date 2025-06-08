import shutil
import pathlib
import argparse
from tqdm import tqdm


def main():
    args = _parse_args()
    root_depth_path = args.root_input_path

    move_camera_02_input_files_and_remove_unecessary(root_input_path=root_depth_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-rip",
        "--root-input-path",
        default=pathlib.Path("./data/kitti/input"),
        type=pathlib.Path,
    )
    return parser.parse_args()


def move_camera_02_input_files_and_remove_unecessary(
    root_input_path: pathlib.Path,
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
    initial_dirs = sorted(root_input_path.iterdir())
    for curr_dir in tqdm(initial_dirs, "Moving files..."):
        date = "_".join(curr_dir.stem.split("_")[:3])  # year_month_day
        if "calib" in str(curr_dir):
            calib_destination_dir = root_input_path / date  # input/year_month_day
            source_dir = (
                curr_dir / date
            )  # year_month_day_drive_xxxx_sync/year_month_day
            shutil.move(source_dir, calib_destination_dir)
        else:
            drive = curr_dir.stem
            image_destination_dir = root_input_path / date / drive / "image_02" / "data"
            source_dir = (
                curr_dir / date / curr_dir.stem / "image_02" / "data"
            )  # year_month_day_drive_xxxx_sync/year_month_day/year_month_day_drive_xxxx_sync
            shutil.move(source_dir, image_destination_dir)

    for dir in initial_dirs:
        shutil.rmtree(dir)


if __name__ == "__main__":
    main()
