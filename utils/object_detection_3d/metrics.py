import torch
from torch import nn

import numpy as np
from scipy.spatial import ConvexHull
from numpy import *

from utils.shared.enums import ObjectDetectionEnum
from utils.object_detection_3d.utils import get_corners

NUM_CLASSES = 4
EPSILON = 1e-5
IOU_THRESHOLD = 0.5
PRED_LABEL_INDEX = 0
PRED_PROBITS_INDEX = 1
GT_BOUNDING_BOX_3D_SLICE = slice(ObjectDetectionEnum.height, None)
PRED_BOUNDING_BOX_3D_SLICE = slice(6, None)


class mAP_BEV(nn.Module):
    higher = True

    def __init__(self):
        super().__init__()
        self.eval()
        self.register_forward_pre_hook(mAP_BEV._extract_relevant_tensor_info)

    @staticmethod
    def _extract_relevant_tensor_info(
        module: nn.Module,
        inputs: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred, gt = inputs
        gt_info = gt["gt_info"].squeeze(0)
        pred_labels = pred[:, 0]
        pred_probits = pred[:, 1]
        pred_boxes_corners = get_corners(pred[:, PRED_BOUNDING_BOX_3D_SLICE])
        gt_label = gt_info[:, ObjectDetectionEnum.object_class]
        gt_boxes_corners = get_corners(gt_info[:, GT_BOUNDING_BOX_3D_SLICE])
        return (pred_labels, pred_probits, pred_boxes_corners), (
            gt_label,
            gt_boxes_corners,
        )

    def forward(
        self, pred: tuple[torch.Tensor, torch.Tensor, torch.Tensor], gt: torch.Tensor
    ) -> torch.Tensor:
        return mAP_3D_BEV_pascal_voc(pred=pred, gt=gt, which="BEV")


class mAP_3D(nn.Module):
    higher = True

    def __init__(self):
        super().__init__()
        self.eval()
        self.register_forward_pre_hook(mAP_BEV._extract_relevant_tensor_info)

    @staticmethod
    def _extract_relevant_tensor_info(
        module: nn.Module,
        inputs: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred, gt = inputs
        gt_info = gt["gt_info"].squeeze(0)
        pred_labels = pred[:, 0]
        pred_probits = pred[:, 1]
        pred_boxes_corners = get_corners(pred[:, PRED_BOUNDING_BOX_3D_SLICE])
        gt_label = gt_info[:, ObjectDetectionEnum.object_class]
        gt_boxes_corners = get_corners(gt_info[:, GT_BOUNDING_BOX_3D_SLICE])
        return (pred_labels, pred_probits, pred_boxes_corners), (
            gt_label,
            gt_boxes_corners,
        )

    def forward(
        self, pred: tuple[torch.Tensor, torch.Tensor, torch.Tensor], gt: torch.Tensor
    ) -> torch.Tensor:
        return mAP_3D_BEV_pascal_voc(pred=pred, gt=gt, which="BEV")


def mAP_3D_BEV_pascal_voc(
    pred: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    gt: tuple[torch.Tensor, torch.Tensor],
    which: str = "3D",
) -> torch.Tensor:
    """
    Although we predict num_classes + background (class 0), we filter it before,
    and we only take into consideration classes from 1 to num_classes.
    This works only for a single image.

    Args:
        - pred: Tuple of (labels, class_probits, pred_boxes).
        - gt: Ground truth object of format (label, top, left, bottom, right).
    """
    pred_labels, class_probits, pred_boxes = pred
    gt_labels, gt_boxes = gt
    device = pred_labels.device
    recall_points_aranged = torch.arange(0, 1.1, 0.1).to(device)
    average_precision_per_class = torch.zeros(NUM_CLASSES - 1).to(device)
    for c in range(1, NUM_CLASSES):
        # Fetch indices of current class
        pred_indices = torch.where(pred_labels == c)[0]
        gt_indices = torch.where(gt_labels == c)[0]

        # Fetch corresponding class information
        pred_boxes_per_class = pred_boxes[pred_indices]
        class_probits_per_class = class_probits[pred_indices]
        gt_boxes_per_class = gt_boxes[gt_indices]

        if pred_boxes_per_class.numel() == 0 or gt_boxes_per_class.numel() == 0:
            # If there are no predictions or no ground truth classes
            # they do not contribute to AP for that class so it will
            # remain zero
            continue

        # Sort by scores descending
        sorted_indices = torch.argsort(class_probits_per_class, descending=True)
        pred_boxes_per_class = pred_boxes_per_class[sorted_indices]
        class_probits_per_class = class_probits_per_class[sorted_indices]

        # Calculate IoU
        iou_per_class = torch.zeros(
            (gt_boxes_per_class.shape[0], pred_boxes_per_class.shape[0])
        )
        for gt_box_index, gt_box_per_class in enumerate(gt_boxes_per_class):
            for pred_box_index, pred_box_per_class in enumerate(pred_boxes_per_class):
                iou_per_class[gt_box_index, pred_box_index] = box3d_iou(
                    corners1=gt_box_per_class.T.cpu().numpy(),
                    corners2=pred_box_per_class.T.cpu().numpy(),
                )[0 if which == "3D" else 1]
        num_gt_boxes = gt_boxes_per_class.shape[0]
        iou_per_class.to(device)
        gt_matched = torch.zeros(num_gt_boxes, dtype=torch.int)
        tp = torch.zeros_like(class_probits_per_class)
        fp = torch.zeros_like(class_probits_per_class)
        for pred_index, pred_box_per_class in enumerate(pred_boxes_per_class):
            possible_gts = torch.where(iou_per_class[:, pred_index] > IOU_THRESHOLD)[0]
            if possible_gts.numel() > 0:
                # If IoU of ground truths and current prediction is larger than threshold
                # choose the largest IoU value, mark it as TP and pair it to corresponding ground truth
                corresponding_gt_pred_ious = iou_per_class[possible_gts, pred_index]
                chosen_gt = torch.argmax(corresponding_gt_pred_ious)
                if gt_matched[chosen_gt] == 0:
                    # If ground truth hasn't been paired, mark it and set the TP flag
                    gt_matched[chosen_gt] = 1
                    tp[pred_index] = 1
                else:
                    # If it has already been paired, all other detections on the same
                    # ground truth are marked as false positives
                    fp[pred_index] = 1
            else:
                # If no IoUs between prediction and ground truths are larger than threshold
                # we set FP flag
                fp[pred_index] = 1

        tp_cumsum = tp.cumsum(dim=-1)
        fp_cumsum = fp.cumsum(dim=-1)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + EPSILON)
        recall = tp_cumsum / (num_gt_boxes + EPSILON)
        # https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173
        average_precision = 0.0
        for t in recall_points_aranged:
            mask = recall >= t
            p_max = precision[mask].max() if mask.any() else torch.zeros(1).to(device)
            average_precision += p_max.item()
        average_precision /= 11.0
        average_precision_per_class[c - 1] = average_precision
    return average_precision_per_class.mean()


# Code from this line to the end is taken from: https://github.com/AlienCat-K/3D-IoU-Python/blob/master/3D-IoU-Python.py
# with small modifications
# 3D IoU caculate code for 3D object detection
# Kent 2018/12


def polygon_clip(subjectPolygon, clipPolygon):
    """Clip a polygon with another polygon.

    Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python

    Args:
      subjectPolygon: a list of (x,y) 2d points, any polygon.
      clipPolygon: a list of (x,y) 2d points, has to be *convex*
    Note:
      **points have to be counter-clockwise ordered**

    Return:
      a list of (x,y) vertex point for the intersection polygon.
    """

    def inside(p):
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

    def computeIntersection():
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

    outputList = subjectPolygon
    cp1 = clipPolygon[-1]

    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]

        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
        if len(outputList) == 0:
            return None
    return outputList


def poly_area(x, y):
    """Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates"""
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def convex_hull_intersection(p1, p2):
    """Compute area of two convex hull's intersection area.
    p1,p2 are a list of (x,y) tuples of hull vertices.
    return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1, p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0


def box3d_vol(corners):
    """corners: (8,3) no assumption on axis direction"""
    a = np.sqrt(np.sum((corners[0, :] - corners[1, :]) ** 2))
    b = np.sqrt(np.sum((corners[1, :] - corners[2, :]) ** 2))
    c = np.sqrt(np.sum((corners[0, :] - corners[4, :]) ** 2))
    return a * b * c


def is_clockwise(p):
    x = p[:, 0]
    y = p[:, 1]
    return np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)) > 0


def box3d_iou(
    corners1: np.ndarray, corners2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU

    todo (kent): add more description on corner points' orders.
    """
    # corner points are in counter clockwise order
    rect1 = [(corners1[i, 0], corners1[i, 2]) for i in range(3, -1, -1)]
    rect2 = [(corners2[i, 0], corners2[i, 2]) for i in range(3, -1, -1)]

    area1 = poly_area(np.array(rect1)[:, 0], np.array(rect1)[:, 1])
    area2 = poly_area(np.array(rect2)[:, 0], np.array(rect2)[:, 1])

    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area / (area1 + area2 - inter_area)
    ymax = min(corners1[0, 1], corners2[0, 1])
    ymin = max(corners1[4, 1], corners2[4, 1])

    inter_vol = inter_area * max(0.0, ymax - ymin)

    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d


# ----------------------------------
# Helper functions for evaluation
# ----------------------------------


def get_3d_box(box_size, heading_angle, center):
    """Calculate 3D bounding box corners from its parameterization.

    Input:
        box_size: tuple of (length,wide,height)
        heading_angle: rad scalar, clockwise from pos x axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box cornders
    """

    def roty(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    R = roty(heading_angle)
    l, w, h = box_size
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + center[0]
    corners_3d[1, :] = corners_3d[1, :] + center[1]
    corners_3d[2, :] = corners_3d[2, :] + center[2]
    corners_3d = np.transpose(corners_3d)
    return corners_3d


if __name__ == "__main__":
    print("------------------")
    # get_3d_box(box_size, heading_angle, center)
    corners_3d_ground = get_3d_box((1, 1, 1), 0, (0, 0, 0))
    corners_3d_predict = get_3d_box((0.5, 0.5, 0.5), 0, (0, 0, 0))
    (IOU_3d, IOU_2d) = box3d_iou(corners_3d_predict, corners_3d_ground)
    print(IOU_3d, IOU_2d)  # 3d IoU/ 2d IoU of BEV(bird eye's view)
