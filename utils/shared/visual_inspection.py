import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes
import dataset.kitti.dataset_utils as KITTIUtils
from dataset.kitti.dataset_utils import TaskEnum
from utils.object_detection.utils import apply_deltas_to_boxes

from utils.object_detection_3d.utils import project_3d_boxes_to_image


def plot_task_gt(task_ground_truth: dict[str, torch.Tensor]):
    tasks = task_ground_truth.keys()
    fig, ax = plt.subplots(
        len(tasks), 1, figsize=(8, 36)
    )  # Note: +1 is for 2D grond truth labels of object detection

    for i, (task, ground_truth) in enumerate(task_ground_truth.items()):
        if task not in list(TaskEnum):  # for projection matrices
            continue
        if task == TaskEnum.object_detection_2d:
            image = (
                task_ground_truth[TaskEnum.input]
                .squeeze(0)
                .permute(1, 2, 0)
                .detach()
                .cpu()
                .numpy()
                .astype(np.uint8)
                .copy()
            )
            ground_truth = ground_truth.squeeze(0).detach().cpu().numpy()
            for object_info in ground_truth:
                bounding_box = np.array(
                    [int(image_coords) for image_coords in object_info[1:]]
                )
                draw_bbox(image, bounding_box)
            ax[i].imshow(image)
            ax[i].set_title(f"Task: {task}")
            ax[i].axis("off")
        elif task == TaskEnum.object_detection_3d:
            image = (
                task_ground_truth[TaskEnum.input]
                .squeeze(0)
                .permute(1, 2, 0)
                .detach()
                .cpu()
                .numpy()
                .astype(np.uint8)
                .copy()
            )
            ground_truth = ground_truth.squeeze(0).detach()
            projection_matrix = (
                task_ground_truth["projection_matrix"].squeeze(0).detach()
            )

            boxes_3d_projected = (
                project_3d_boxes_to_image(ground_truth[:, 1:], projection_matrix)
                .cpu()
                .numpy()
                .astype(np.int16)
            )
            for bounding_box_3d in boxes_3d_projected:
                draw_3d_bbox(image, bounding_box_3d.T)

            ax[i + 1].imshow(image)
            ax[i + 1].set_title(f"Task: {task}")
            ax[i + 1].axis("off")
        else:
            image = (
                ground_truth.squeeze(0)
                .permute(1, 2, 0)
                .detach()
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
        ax[i].imshow(image)
        ax[i].set_title(f"Task: {task}")
        ax[i].axis("off")

    def on_key(event):
        plt.close()

    plt.tight_layout()
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


def draw_bbox(image: np.ndarray, bbox: np.ndarray):
    left, top, right, bottom = bbox
    # Note: cv2 coordinate system starts at the bottom left of an image
    cv2.line(image, (left, top), (right, top), color=(255, 0, 0), thickness=1)
    cv2.line(image, (left, top), (left, bottom), color=(255, 0, 0), thickness=1)
    cv2.line(image, (right, top), (right, bottom), color=(255, 0, 0), thickness=1)
    cv2.line(image, (left, bottom), (right, bottom), color=(255, 0, 0), thickness=1)


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


def plot_object_detection_predictions_2d(
    input_image: torch.Tensor,
    predicted_bounding_boxes: torch.Tensor,
    ground_truth: torch.Tensor,
    save_name,
):
    fig, ax = plt.subplots(1, 1, figsize=(20, 16))
    input_image = input_image.squeeze(0).detach().cpu().to(torch.uint8).clone()
    pred_class_probits, pred_boxes, pred_labels = predicted_bounding_boxes
    ground_truth_boxes = ground_truth.squeeze(0).cpu().to(torch.int64)[:, 1:]
    groun_truth_labels = ground_truth.squeeze(0).cpu().to(torch.int64)[:, :1]
    red = (255, 0, 0)
    green = (0, 255, 0)
    if pred_boxes.shape[0] == 1:
        return
    input_image = draw_bounding_boxes(
        image=input_image,
        boxes=pred_boxes.detach().cpu().squeeze().to(torch.int64),
        labels=[str(int(pred_label.item())) for pred_label in pred_labels],
        colors=red,
    )
    input_image = draw_bounding_boxes(
        image=input_image,
        boxes=ground_truth_boxes,
        # labels=[str(int(gt_label.item())) for gt_label in groun_truth_labels],
        colors=green,
        fill=True,
    )
    ax.imshow(input_image.permute(1, 2, 0))
    plt.tight_layout()
    plt.savefig(save_name)
