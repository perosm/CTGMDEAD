import torch
from torch import nn
from torchvision.ops import box_iou, clip_boxes_to_image
from utils.object_detection.utils import (
    apply_deltas_to_boxes,
    get_deltas_from_bounding_boxes,
)


class RPNClassificationAndRegressionLoss(nn.Module):

    def __init__(self, regularization_factor: float = 10.0) -> None:
        super().__init__()
        self.n_cls = 256 * 4  # * 4 for each fpn output
        self.iou_negative_threshold = 0.3
        self.iou_positive_threshold = 0.5
        self.regularization_factor = 10.0
        self.classification_loss_fn = nn.BCELoss(reduction="mean").to("cuda")
        self.regression_loss_fn = nn.SmoothL1Loss(reduction="mean").to("cuda")
        self.register_forward_pre_hook(self._extract_relevant_tensor_info)

    @staticmethod
    def _extract_relevant_tensor_info(
        module: nn.Module,
        inputs: tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred_info, gt = inputs
        anchors, all_objectness_probits, pred_deltas, _, _ = pred_info["rpn"]
        gt_bounding_box = gt[..., 1:]

        return anchors, all_objectness_probits, pred_deltas, gt_bounding_box

    def forward(
        self,
        anchors: torch.Tensor,
        objectness_probits: torch.Tensor,
        pred_deltas: torch.Tensor,
        gt_bounding_box: torch.Tensor,
    ) -> torch.Tensor:
        """
        https://arxiv.org/pdf/1506.01497
        Calculates IoU between ground truth and predicted bounding boxes.
        Boxes with IoU > positive_threshold are positive ones and boxes
        with IoU < negative_threshold are negative. Others are not considered.
        We take n_cls / 2 positive boxes (if there are that many) and the rest are negative.

        Args:
            anchors: Predicted bounding boxes by the RPN of shape (num_anchors, 4). # TODO: make it work for N > 1
            objectness_probits: Objectness score for bounding boxes (N, num_anchors).
            pred_deltas: Predicted anchor offsets (N, num_anchors, 4).
            gt_bounding_box: Ground truth object bounding boxes (num_objects_in_frame, 4).

        Returns:
            Binary Cross Entropy Loss between positive and negative boxes.
        """
        gt_bounding_box = gt_bounding_box.squeeze(0)
        pred_deltas = pred_deltas.squeeze(0)
        iou_matrix = box_iou(boxes1=gt_bounding_box, boxes2=anchors)

        # Finding all positive and negative anchors
        pos_rows, pos_cols = torch.where(iou_matrix > self.iou_positive_threshold)
        neg_rows, neg_cols = torch.where(iou_matrix < self.iou_negative_threshold)

        # Force > 1 positive example
        gt_max_iou, gt_argmax = iou_matrix.max(dim=1)
        pos_cols = torch.cat([pos_cols, gt_argmax])
        pos_rows = torch.cat(
            [pos_rows, torch.arange(len(gt_argmax), device=gt_argmax.device)]
        )

        # Removing duplicates
        unique_pairs = torch.unique(torch.stack([pos_rows, pos_cols]), dim=1)
        pos_rows, pos_cols = unique_pairs[0], unique_pairs[1]

        # Balanced sampling
        num_pos = min(self.n_cls // 2, pos_cols.numel())
        num_neg = min(self.n_cls - num_pos, neg_cols.numel())

        # Random shuffling
        pos_indices = torch.randperm(pos_cols.shape[0])
        pos_cols = pos_cols[pos_indices[:num_pos]]
        pos_rows = pos_rows[pos_indices[:num_pos]]

        neg_indices = torch.randperm(neg_cols.shape[0])
        neg_cols = neg_cols[neg_indices[:num_neg]]
        neg_rows = neg_rows[neg_indices[:num_neg]]

        pos = objectness_probits[:, pos_cols]
        neg = objectness_probits[:, neg_cols]

        pred_classification = torch.cat([pos, neg], dim=1)
        gt_classification = torch.cat(
            [torch.ones_like(pos), torch.zeros_like(neg)], dim=1
        ).to(pred_classification.device)

        gt_bounding_box = gt_bounding_box[pos_rows, :]
        pos_anchors = anchors[pos_cols, :]
        gt_deltas = get_deltas_from_bounding_boxes(
            gt_bounding_box=gt_bounding_box, pred_bounding_box=pos_anchors
        )
        pred_deltas = pred_deltas[pos_cols, :]

        classification_loss = self.classification_loss_fn(
            pred_classification, gt_classification
        )
        regression_loss = self.regression_loss_fn(pred_deltas, gt_deltas)
        print("RPN", classification_loss, regression_loss)
        return classification_loss + self.regularization_factor * regression_loss


class RCNNCrossEntropyAndRegressionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.neg_iou_threshold = 0.3
        self.pos_iou_threshold = 0.5
        self.n_cls = 256
        self.num_classes = 3
        self.classification_loss_fn = nn.CrossEntropyLoss(reduction="mean")
        self.regression_loss_fn = nn.SmoothL1Loss(reduction="mean")
        self.register_forward_pre_hook(self._extract_relevant_tensor_info)

    @staticmethod
    def _extract_relevant_tensor_info(
        module: nn.Module,
        inputs: tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred_info, gt = inputs
        pred_class_logits, proposals, deltas = pred_info["faster-rcnn"]
        gt_class, gt_bounding_boxes = gt[..., 0], gt[..., 1:]

        return pred_class_logits, proposals, deltas, gt_class, gt_bounding_boxes

    def forward(
        self,
        pred_class_logits: torch.Tensor,
        proposals: torch.Tensor,
        pred_deltas: torch.Tensor,
        gt_class: torch.Tensor,
        gt_bounding_boxes: torch.Tensor,
    ) -> torch.Tensor:
        gt_bounding_boxes = gt_bounding_boxes.squeeze(0)
        gt_class = gt_class.squeeze(0).to(torch.int64)
        num_proposals = pred_class_logits.shape[0]
        num_gt = gt_class.shape[0]
        iou_matrix = box_iou(boxes1=gt_bounding_boxes, boxes2=proposals)

        # force atleast one predictions per ground truth
        max_ious, best_prop_per_gt_indices = iou_matrix.max(dim=1)
        forced_gt_indices = torch.arange(
            best_prop_per_gt_indices.shape[0], device=best_prop_per_gt_indices.device
        )

        # object_rows represents the gt object corresponding to the predicted object
        # which is represented by the object_cols in the iou_matrix if IoU > threshold
        pos_gt_indices, pos_prop_indices = torch.where(
            iou_matrix > self.pos_iou_threshold
        )

        # combining both forced and condition satisfying indices
        pos_gt_indices = torch.cat([forced_gt_indices, pos_gt_indices], dim=0)
        pos_prop_indices = torch.cat(
            [best_prop_per_gt_indices, pos_prop_indices], dim=0
        )

        # TODO: limit number of predictions per gt ?

        # negative indices
        neg_gt_indices, neg_prop_indices = torch.where(
            iou_matrix < self.neg_iou_threshold
        )

        # balanced sampling
        num_pos = min(self.n_cls // 2, pos_prop_indices.numel())
        num_neg = self.n_cls - num_pos

        pos_permute = torch.randperm(pos_prop_indices.numel())
        neg_permute = torch.randperm(neg_prop_indices.numel())
        pos_prop_indices = pos_prop_indices[pos_permute][:num_pos]
        neg_prop_indices = neg_prop_indices[neg_permute][:num_neg]
        pos_gt_indices = pos_gt_indices[pos_permute][:num_pos]
        neg_gt_indices = neg_gt_indices[neg_permute][:num_neg]

        gt_indices = torch.cat([pos_gt_indices, neg_gt_indices], dim=0)
        pred_indices = torch.cat([pos_prop_indices, neg_prop_indices], dim=0)

        # filter boxes if they satisfy IoU condition
        pred_class_logits = pred_class_logits[pred_indices]
        proposals = proposals[pos_prop_indices]
        pred_deltas = pred_deltas[pos_prop_indices]

        # Fetching predicted per class deltas
        pred_class_indices = pred_class_logits.argmax(dim=1).to(torch.int64)

        pred_deltas = pred_deltas.view(-1, self.num_classes, 4)
        pred_per_class_deltas = pred_deltas[
            torch.arange(pred_deltas.shape[0]), pred_class_indices[:num_pos], :
        ]  # for regression loss we use just positive indices
        pred_bounding_box = apply_deltas_to_boxes(
            boxes=proposals, deltas=pred_per_class_deltas
        )

        gt_classes_per_pred = gt_class[gt_indices]
        gt_bounding_boxes_per_pred = gt_bounding_boxes[pos_gt_indices, :]
        gt_deltas = get_deltas_from_bounding_boxes(
            gt_bounding_box=gt_bounding_boxes_per_pred,
            pred_bounding_box=pred_bounding_box,
        )
        classification_loss = self.classification_loss_fn(
            pred_class_logits, gt_classes_per_pred.to(torch.int64)
        )
        regression_loss = self.regression_loss_fn(pred_per_class_deltas, gt_deltas)
        print("RCNN", classification_loss, regression_loss)
        return classification_loss + regression_loss
