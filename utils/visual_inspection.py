import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import dataset.dataset_utils as KITTIUtils
from matplotlib import patches
from dataset.dataset_utils import TaskEnum


def plot_task_gt(task_groundtruth: dict[str, torch.Tensor]):
    tasks = task_groundtruth.keys()
    fig, ax = plt.subplots(len(tasks), 1, figsize=(8, 36))

    for i, (task, groundtruth) in enumerate(task_groundtruth.items()):
        if task == TaskEnum.object_detection_3d:
            img = (
                task_groundtruth[TaskEnum.input]
                .squeeze(0)
                .permute(1, 2, 0)
                .detach()
                .cpu()
                .numpy()
                .astype(np.int16)
            )
            groundtruth = groundtruth.squeeze(0).detach().cpu().numpy()
            for object_info in groundtruth:
                left, top, right, bottom = object_info[4:8].astype(np.int16)
                # changing start of axes
                # new_top = abs(KITTIUtils.NEW_H - top)
                # new_bottom = abs(KITTIUtils.NEW_H - bottom)
                width = left - right
                height = top - bottom
                print(object_info[4:8].astype(np.int16))
                #  rect = patches.Rectangle(
                #     xy=(left, KITTIUtils.KITTI_H - top - height),
                #     width=width,
                #     height=height,
                #     linewidth=2,
                #     edgecolor="r",
                #     facecolor="none",
                # )
                # ax[i].add_patch(rect)
                img[top:bottom, left, :] = (255, 0, 0)  # left line
                img[top:bottom, right, :] = (255, 0, 0)  # right line
                img[top, left:right, :] = (255, 0, 0)  # top line
                img[bottom, left:right, :] = (255, 0, 0)  # bottom line

        else:
            img = (
                groundtruth.squeeze(0)
                .permute(1, 2, 0)
                .detach()
                .cpu()
                .numpy()
                .astype(np.int16)
            )
        ax[i].imshow(img)
        ax[i].set_title(f"Task: {task}")
        ax[i].axis("off")

    def on_key(event):
        plt.close()

    plt.tight_layout()
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


def draw_bbox(image: np.ndarray, bbox: np.ndarray):
    left, top, right, bottom = bbox
    # Note: cv2 plots coordinates differently
    cv2.line(image, (left, top), (right, top), color=(255, 0, 0), thickness=1)
    cv2.line(image, (left, top), (left, bottom), color=(255, 0, 0), thickness=1)
    cv2.line(image, (right, top), (right, bottom), color=(255, 0, 0), thickness=1)
    cv2.line(image, (left, bottom), (right, bottom), color=(255, 0, 0), thickness=1)


def project_3d_bbox_to_image(bbox_3d: np.ndarray, projection_matrix: np.ndarray):
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


def draw_3d_bbox(image_3d_bboxes: np.ndarray, projected_points: np.ndarray):
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
