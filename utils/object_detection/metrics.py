import torch
from torch import nn

from torchvision.ops import box_iou
from utils.shared.enums import ObjectDetectionEnum

NUM_CLASSES = 4
EPSILON = 1e-5
IOU_THRESHOLD = 0.5
PRED_LABEL_INDEX = 0
PRED_PROBITS_INDEX = 1
PRED_BOUNDING_BOX_2D_SLICE = slice(2, 6)
GT_BOUNDING_BOX_2D_SLICE = slice(
    ObjectDetectionEnum.box_2d_left, ObjectDetectionEnum.box_2d_bottom + 1
)


class mAP(nn.Module):
    higher = True

    def __init__(self):
        super().__init__()
        self.eval()
        self.register_forward_pre_hook(mAP._extract_relevant_tensor_info)

    @staticmethod
    def _extract_relevant_tensor_info(
        module: nn.Module,
        inputs: tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred, gt = inputs
        gt_info = gt["gt_info"].squeeze(0)

        pred_label = pred[:, PRED_LABEL_INDEX]
        pred_probits = pred[:, PRED_PROBITS_INDEX]
        pred_box_2d = pred[:, PRED_BOUNDING_BOX_2D_SLICE]
        gt_label = gt_info[:, ObjectDetectionEnum.object_class]
        gt_box_2d = gt_info[:, GT_BOUNDING_BOX_2D_SLICE]

        return (pred_label, pred_probits, pred_box_2d), (gt_label, gt_box_2d)

    def forward(
        self,
        pred: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        gt: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        return mAP_pascal_voc(pred=pred, gt=gt)


def mAP_pascal_voc(
    pred: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    gt: tuple[torch.Tensor, torch.Tensor],
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
        for t in recall_points_aranged:
            mask = recall >= t
            p_max = precision[mask].max() if mask.any() else torch.zeros(1).to(device)
            average_precision += p_max.item()
        average_precision /= 11.0
        average_precision_per_class[c - 1] = average_precision
    return average_precision_per_class.mean()
