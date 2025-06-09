import torch
from torch import nn

from torchvision.ops import box_iou

NUM_CLASSES = 4
EPSILON = 1e-5
IOU_THRESHOLD = 0.5


class mAP(nn.Module):
    def __init__(self):
        super().__init__()
        self.eval()

    def forward(
        self, pred: tuple[torch.Tensor, torch.Tensor, torch.Tensor], gt: torch.Tensor
    ) -> torch.Tensor:
        return mAP_pascal_voc(pred=pred, gt=gt)


def mAP_pascal_voc(
    pred: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    gt: torch.Tensor,
) -> torch.Tensor:
    """
    Although we predict num_classes + background (class 0), we filter it before,
    and we only take into consideration classes from 1 to num_classes.
    This works only for a single image.

    Args:
        - pred: Tuple of (class_probits, pred_boxes, labels).
        - gt: Ground truth object of format (label, top, left, bottom, right).
    """
    gt = gt.squeeze(0)
    class_probits, pred_boxes, pred_labels = pred
    gt_labels, gt_boxes = gt[:, 0], gt[:, 1:]
    average_precision_per_class = torch.zeros(NUM_CLASSES - 1)
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
        iou_per_class = box_iou(boxes1=gt_boxes_per_class, boxes2=pred_boxes_per_class)
        num_gt_boxes = gt_boxes_per_class.shape[0]

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
        for r in torch.arange(0, 1.1, 0.1):
            mask = recall >= r
            if mask.any():
                p = precision[mask].max()
            else:
                p = 0.0
            ap += p
        ap /= 11.0

        average_precision_per_class[c - 1] = average_precision
    return average_precision_per_class.mean()
