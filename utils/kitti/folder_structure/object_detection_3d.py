import shutil
import argparse
import pathlib
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils.visual_inspection import project_3d_bbox_to_image, draw_3d_bbox, draw_bbox


def main():
    args = _parse_args()

    input_images_dir = args.input_images_dir
    mapping_path = args.devkit_object_detection_3d_mapping
    ground_truth_dir = args.label_object_detection_3d
    calibration_dir = args.object_detection_3d_calibration
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
        # input_image_path = input_images_dir / subfolders / f"{frame}.png"
        # visual_inspection(input_image_path, ground_truth_path, calibration_path)
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
            calibration_destination_dir / f"{frame}.{calibration_path.suffix}"
        )
    breakpoint()
    shutil.rmtree(mapping_path.parent)
    shutil.rmtree(ground_truth_dir.parent.parent)
    shutil.rmtree(calibration_dir.parent.parent)
    num_files = sum(1 for _ in destination_root_folder.rglob("*.txt") if _.is_file())

    print(f"Number of files: {num_files / 2}")
    print(f"Number of mappings: {len(mapping_info_indexes)}")


def visual_inspection(
    input_image_path: pathlib.Path,
    object_detection_info_path: pathlib.Path,
    object_detection_calibration_path: pathlib.Path,
):
    image = cv2.cvtColor(cv2.imread(input_image_path), cv2.COLOR_BGR2RGB)
    objects_info = _read_objects_info(object_detection_info_path)
    projection_matrix = _read_projection_matrix(object_detection_calibration_path)
    fig, ax = plt.subplots(2, 1)
    image_2d_bboxes = image.copy()
    image_3d_bboxes = image.copy()
    for object_info in objects_info:
        bbox_2d = np.array(
            [int(float(image_coords)) for image_coords in object_info[4:8]]
        )
        bbox_3d = np.array([float(world_coords) for world_coords in object_info[8:15]])
        draw_bbox(image_2d_bboxes, bbox_2d)
        projected_points = project_3d_bbox_to_image(bbox_3d, projection_matrix)
        draw_3d_bbox(image_3d_bboxes, projected_points)

    ax[0].imshow(image_2d_bboxes)
    ax[1].imshow(image_3d_bboxes)

    def on_key():
        plt.close()

    plt.tight_layout()
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


def _read_objects_info(object_detection_info_path: pathlib.Path):
    with open(object_detection_info_path, "r") as file:
        objects_info = [line.strip().split() for line in file.readlines()]

    return objects_info


def _read_projection_matrix(object_detection_calibration_path: pathlib.Path):
    with open(object_detection_calibration_path, "r") as file:
        last_row = np.array([[0, 0, 0, 1]])
        projection_matrix = np.vstack(
            (
                np.array(
                    [
                        float(number)
                        for number in file.readlines()[2].split(": ")[1].strip().split()
                    ]
                ).reshape(3, 4),
                last_row,
            )
        )

    return projection_matrix


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
        "--label-object-detection-3d",
        type=pathlib.Path,
        default=pathlib.Path(
            "./data/kitti/data_object_label_2/training/label_2"
        ),  # left camera images
        help="Path to 3d object detection ground truth.",
    )
    parser.add_argument(
        "-odc",
        "--object-detection-3d-calibration",
        type=pathlib.Path,
        default=pathlib.Path(
            "./data/kitti/data_object_calib/training/calib"
        ),  # left camera images
        help="Path to 3d object detection calibration files.",
    )
    parser.add_argument(
        "-drf",
        "--destination-root-folder",
        type=pathlib.Path,
        default=pathlib.Path("./data/kitti/object_detection_3d"),
        help="Destination root folder for all the corresponding files.",
    )
    parser.add_argument(
        "-dodm",
        "--devkit-object-detection-3d-mapping",
        type=pathlib.Path,
        default=pathlib.Path("./data/kitti/devkit_object/mapping"),
        help="Path to mapping folder. Contains information how to map labels to raw data.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
