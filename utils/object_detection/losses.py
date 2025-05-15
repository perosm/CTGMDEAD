import torch
from torch import nn
from torchvision.ops import box_iou, clip_boxes_to_image
from utils.object_detection.utils import (
    apply_deltas_to_boxes,
    get_deltas_from_bounding_boxes,
)
from torchvision.ops import sigmoid_focal_loss


class RPNClassificationAndRegressionLoss(nn.Module):

    def __init__(self, regularization_factor: float = 10.0) -> None:
        super().__init__()
        self.n_cls = 256
        self.positives_ratio = 0.25
        self.negatives_ratio = 1 - self.positives_ratio
        self.iou_negative_threshold = 0.3
        self.iou_positive_threshold = 0.7
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
        all_anchors, all_objectness_probits, all_pred_deltas, _, _ = pred_info["rpn"]
        gt_bounding_box = gt[..., 1:].squeeze(0)

        return (
            all_anchors,
            all_objectness_probits,
            all_pred_deltas,
            gt_bounding_box,
        )

    def forward(
        self,
        all_anchors: torch.Tensor,
        all_objectness_probits: torch.Tensor,
        all_pred_deltas: torch.Tensor,
        gt_bounding_box: torch.Tensor,
    ) -> torch.Tensor:
        """
        https://arxiv.org/pdf/1506.01497
        Calculates IoU between ground truth and anchor boxes.
        Anchors with IoU > positive_threshold are positive ones and anchors
        with IoU < negative_threshold are negative. Others are not considered.
        We take n_cls / 2 positives (if there are that many) and the rest are negatives.
        We calculate regression loss between paired anchors and gt for positives
        by calculating regression deltas for ground truths
        and classification loss between gt and positives + negatives.

        Args:
            anchors: Predicted bounding boxes by the RPN of shape (num_anchors, 4).
            objectness_probits: Objectness score for bounding boxes (num_anchors).
            pred_deltas: Predicted anchor offsets (num_anchors, 4).
            gt_bounding_box: Ground truth object bounding boxes (num_objects_in_frame, 4).

        Returns:
            Binary Cross Entropy Loss between positive and negative boxes.
        """
        # Computing IoU between gt and all anchors
        iou_matrix = box_iou(boxes1=gt_bounding_box, boxes2=all_anchors)

        # For each anchor fing gt with highest IoU
        max_iou_per_anchor, gt_indices = iou_matrix.max(dim=0)

        # For each gt find anchor with highest IoU
        gt_best_anchor_indices = iou_matrix.argmax(dim=1)

        # For positive anchors we take all anchors whose
        # IoU value > iou_positive_threshold
        # and we force always at least one anchor per gt
        positive_mask = max_iou_per_anchor > self.iou_positive_threshold
        positive_mask[gt_best_anchor_indices] = True

        # For negative anchors we take all anchors whose
        # IoU value < iou_negative_threshold
        negative_mask = (
            max_iou_per_anchor < self.iou_negative_threshold
        ) & ~positive_mask  # for preventing overlap

        pos_anchor_indices = torch.where(positive_mask)[0]
        neg_anchor_indices = torch.where(negative_mask)[0]

        pos_gt_indices = gt_indices[pos_anchor_indices]

        # Balanced sampling
        num_pos = self.positives_ratio * self.n_cls
        num_pos = min(num_pos, pos_anchor_indices.numel())
        num_neg = int(num_pos / self.positives_ratio * self.negatives_ratio)
        num_neg = min(num_neg, neg_anchor_indices.numel())

        # Random shuffling
        shuffled_pos = torch.randperm(pos_anchor_indices.shape[0])
        pos_sample = pos_anchor_indices[shuffled_pos[:num_pos]]

        shuffled_neg = torch.randperm(neg_anchor_indices.shape[0])
        neg_sample = neg_anchor_indices[shuffled_neg[:num_neg]]

        # Classification
        pred_pos = all_objectness_probits[pos_sample]
        pred_neg = all_objectness_probits[neg_sample]
        pred_classification = torch.cat([pred_pos, pred_neg])
        gt_classification = torch.cat(
            [torch.ones_like(pred_pos), torch.zeros_like(pred_neg)], dim=0
        ).to(pred_classification.device)
        classification_loss = self.classification_loss_fn(
            pred_classification, gt_classification
        )
        # classification_loss = sigmoid_focal_loss(
        #     inputs=pred_classification,
        #     targets=gt_classification,
        #     alpha=1 - (num_pos / num_neg),
        #     reduction="mean",
        # )
        # Regression
        gt_indices_sample = pos_gt_indices[shuffled_pos[:num_pos]]
        pos_gt = gt_bounding_box[gt_indices_sample]
        pos_anchors = all_anchors[pos_sample]
        gt_deltas = get_deltas_from_bounding_boxes(
            reference_boxes=pos_gt,
            predicted_boxes=pos_anchors,  # TODO: other way around?
        )
        pred_deltas_sampled = all_pred_deltas[pos_sample, :]
        regression_loss = self.regularization_factor * self.regression_loss_fn(
            pred_deltas_sampled, gt_deltas
        )

        return classification_loss + regression_loss


class RCNNCrossEntropyAndRegressionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.positives_ratio = 0.25
        self.negatives_ratio = 1 - self.positives_ratio
        self.iou_positive_threshold = 0.5
        self.iou_negative_threshold = 0.2
        self.n_cls = 256
        self.num_classes = 3
        self.regularization_factor = 10.0
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
        gt_class, gt_bounding_boxes = gt[..., 0], gt[..., 1:].squeeze(0)

        return pred_class_logits, proposals, deltas, gt_class, gt_bounding_boxes

    def forward(
        self,
        pred_class_logits: torch.Tensor,
        proposals: torch.Tensor,
        pred_deltas: torch.Tensor,
        gt_class: torch.Tensor,
        gt_bounding_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """ """
        # Computing IoU between gt and all anchors
        iou_matrix = box_iou(boxes1=gt_bounding_boxes, boxes2=proposals)

        # For each anchor fing gt with highest IoU
        max_iou_per_proposal, gt_indices = iou_matrix.max(dim=0)

        # For each gt find anchor with highest IoU
        gt_best_proposal_indices = iou_matrix.argmax(dim=1)

        # For positive proposals we take all anchors whose
        # IoU > iou_positive_threshold
        # and we force always at least one anchor per gt
        positive_mask = max_iou_per_proposal > self.iou_positive_threshold
        positive_mask[gt_best_proposal_indices] = True

        # For negative anchors we take all anchors whose
        # IoU value < iou_negative_threshold
        negative_mask = (
            max_iou_per_proposal < self.iou_negative_threshold
        ) & ~positive_mask

        pos_proposal_indices = torch.where(positive_mask)[0]
        neg_proposal_indices = torch.where(negative_mask)[0]

        pos_gt_indices = gt_indices[pos_proposal_indices]
        neg_gt_indices = gt_indices[neg_proposal_indices]

        # Balanced sampling
        num_pos = self.positives_ratio * self.n_cls
        num_pos = min(num_pos, pos_proposal_indices.numel())
        num_neg = int(num_pos / self.positives_ratio * self.negatives_ratio)
        num_neg = min(num_neg, neg_proposal_indices.numel())

        # Random shuffling
        shuffled_pos = torch.randperm(pos_proposal_indices.shape[0])
        pos_sample = pos_proposal_indices[shuffled_pos[:num_pos]]

        shuffled_neg = torch.randperm(neg_proposal_indices.shape[0])
        neg_sample = neg_proposal_indices[shuffled_neg[:num_neg]]

        pos_gt = pos_gt_indices[shuffled_pos[:num_pos]]
        neg_gt = neg_gt_indices[shuffled_neg[:num_neg]]

        # Classification
        pred_pos = pred_class_logits[pos_sample]
        pred_neg = pred_class_logits[neg_sample]
        pred_classification = torch.cat([pred_pos, pred_neg])

        gt_classification = ...
        classification_loss = self.classification_loss_fn

        # Regression
        regression_loss = ...
        return classification_loss + regression_loss
