import shutil
import argparse
import pathlib
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import axes


def main():
    args = _parse_args()

    input_image_paths = sorted(
        [file for file in args.data_object_detection_3d.iterdir()]
    )
    ground_truth_paths = sorted(
        [file for file in args.label_object_detection_3d.iterdir()]
    )
    calibration_paths = sorted(
        [file for file in args.object_detection_3d_calibration.iterdir()]
    )
    for input_image_path, ground_truth_path, calibration_path in zip(
        input_image_paths, ground_truth_paths, calibration_paths
    ):
        visual_inspection(input_image_path, ground_truth_path, calibration_path)
        # full_source_path_name.rename(full_destination_path_name / image_name)

    # shutil.rmtree(args.data_object_detection_3d.parent.parent)
    # shutil.rmtree(args.devkit_object_detection_3d_mapping.parent.parent)
    # num_files = sum(1 for _ in destination_root_dir_name.rglob("*") if _.is_file())

    # print(f"Number of files: {num_files}")
    # print(f"Number of mappings: {len(lines)}")


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
        _draw_bbox(image_2d_bboxes, bbox_2d)
        projected_points = _project_3d_bbox_to_image(bbox_3d, projection_matrix)
        _draw_3d_bbox(image_3d_bboxes, projected_points)

    ax[0].imshow(image_2d_bboxes)
    ax[1].imshow(image_3d_bboxes)

    def on_key(event):
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


def _draw_bbox(image: np.ndarray, bbox: np.ndarray):
    left, top, right, bottom = bbox
    # Note: cv2 plots coordinates differently
    cv2.line(image, (left, top), (right, top), color=(255, 0, 0), thickness=1)
    cv2.line(image, (left, top), (left, bottom), color=(255, 0, 0), thickness=1)
    cv2.line(image, (right, top), (right, bottom), color=(255, 0, 0), thickness=1)
    cv2.line(image, (left, bottom), (right, bottom), color=(255, 0, 0), thickness=1)


def _project_3d_bbox_to_image(bbox_3d: np.ndarray, projection_matrix: np.ndarray):
    h, w, l = bbox_3d[0:3]
    x, y, z = bbox_3d[3:6]
    rotation_y = bbox_3d[6]
    R = np.array(
        [
            [np.cos(rotation_y), 0, np.sin(rotation_y)],
            [0, 1, 0],
            [-np.sin(rotation_y), 0, np.cos(rotation_y)],
        ]
    )
    x_corners = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
    y_corners = np.array([0, 0, 0, 0, -h, -h, -h, -h])
    z_corners = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])
    corners = np.vstack([x_corners, y_corners, z_corners])
    corners = R @ corners
    corners[0, :] += x
    corners[1, :] += y
    corners[2, :] += z
    corners_homogeneous = np.vstack([corners, np.ones(shape=(1, corners.shape[1]))])
    projections = projection_matrix @ corners_homogeneous
    projections = projections[:2, :] / projections[2, :]

    return projections.T.astype(np.int16)


def _draw_3d_bbox(image_3d_bboxes: np.ndarray, projected_points: np.ndarray):
    edges = [
        # bottom
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        # top
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        # vertical
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for start, end in edges:
        start_point = projected_points[start]
        end_point = projected_points[end]
        cv2.line(
            image_3d_bboxes, start_point, end_point, color=(255, 0, 0), thickness=1
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dod",
        "--data-object-detection-3d",
        type=pathlib.Path,
        default=pathlib.Path(
            "./data/kitti/data_object_image_2/training/image_2"
        ),  # left camera images
        help="Path to 3d object detection input images.",
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
    return parser.parse_args()


if __name__ == "__main__":
    main()
