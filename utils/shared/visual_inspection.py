import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes
import dataset.dataset_utils as KITTIUtils
from dataset.dataset_utils import TaskEnum
from utils.object_detection.utils import apply_deltas_to_boxes


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
            ground_truth = ground_truth.detach().cpu().numpy()
            projection_matrix = (
                task_ground_truth["projection_matrix"].squeeze(0).detach().cpu().numpy()
            )
            for bounding_box_3d in ground_truth:

                bbox_3d_projected = project_3d_bbox_to_image(
                    bounding_box_3d, projection_matrix
                )
                draw_3d_bbox(image, bbox_3d_projected)

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


def _read_object_detection_gt_info(object_info: np.ndarray) -> np.ndarray:
    bbox_2d = np.array([int(float(image_coords)) for image_coords in object_info[4:8]])
    bbox_3d = np.array([float(world_coords) for world_coords in object_info[8:15]])

    return bbox_2d, bbox_3d


def draw_bbox(image: np.ndarray, bbox: np.ndarray):
    left, top, right, bottom = bbox
    # Note: cv2 coordinate system starts at the bottom left of an image
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


def plot_object_detection_predictions_2d(
    input_image: torch.Tensor,
    predicted_bounding_boxes: torch.Tensor,
    ground_truth_boxes: torch.Tensor,
    save_name,
):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 16))
    image_rpn = input_image.squeeze(0).detach().cpu().to(torch.uint8).clone()
    image_faster_rcnn = input_image.squeeze(0).detach().cpu().to(torch.uint8).clone()
    (
        anchors,
        all_objectness_scores,
        anchor_deltas,
        filtered_objectness_scores,
        proposals,
    ) = predicted_bounding_boxes["rpn"]
    _, top_k_boxes_rpn_indices = torch.topk(all_objectness_scores, k=7)
    anchors = anchors[top_k_boxes_rpn_indices]
    anchor_deltas = anchor_deltas[top_k_boxes_rpn_indices]
    anchors = apply_deltas_to_boxes(anchors, anchor_deltas)
    anchors = anchors.detach().cpu()

    pred_class_logits, filtered_proposals, proposal_deltas = predicted_bounding_boxes[
        "faster-rcnn"
    ]
    pred_class_indices = pred_class_logits.argmax(dim=1).to(torch.int64)
    pred_per_class_deltas = proposal_deltas.view(-1, 3, 4)[
        torch.arange(pred_class_indices.shape[0]), pred_class_indices, :
    ]
    pred_bounding_box = apply_deltas_to_boxes(
        boxes=filtered_proposals, deltas=pred_per_class_deltas
    )
    pred_bounding_box = pred_bounding_box.detach()

    ground_truth_boxes = ground_truth_boxes.squeeze(0).cpu().to(torch.int64)[:, 1:]
    red = (255, 0, 0)
    green = (0, 255, 0)
    image_rpn = draw_bounding_boxes(
        image=image_rpn,
        boxes=anchors.squeeze().to(torch.int64),
        colors=red,
    )
    image_rpn = draw_bounding_boxes(
        image=image_rpn,
        boxes=ground_truth_boxes,
        colors=green,
    )
    image_faster_rcnn = draw_bounding_boxes(
        image=image_faster_rcnn,
        boxes=pred_bounding_box.squeeze().to(torch.int64),
        colors=red,
    )
    image_faster_rcnn = draw_bounding_boxes(
        image=image_faster_rcnn,
        boxes=ground_truth_boxes,
        colors=green,
    )
    ax1.imshow(image_rpn.permute(1, 2, 0).numpy())
    ax2.imshow(image_faster_rcnn.permute(1, 2, 0).numpy())
    plt.tight_layout()
    plt.savefig(save_name)
