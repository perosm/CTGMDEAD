import torch
from torch import nn
from torchvision.ops import box_iou


class RPNClassificationAndRegressionLoss(nn.Module):

    def __init__(self, regularization_factor: float = 10.0) -> None:
        super().__init__()
        self.n_cls = 256
        self.iou_negative_threshold = 0.1
        self.iou_positive_threshold = 0.4
        self.regularization_factor = regularization_factor
        self.classification_loss = nn.BCEWithLogitsLoss(reduction="mean")
        self.regression_loss = nn.SmoothL1Loss(reduction="mean")
        self.register_forward_pre_hook(self._extract_relevant_tensor_info)

    @staticmethod
    def _extract_relevant_tensor_info(
        module: nn.Module,
        inputs: tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred_info, gt = inputs
        objectness_score, proposals = pred_info["rpn"]
        gt_bounding_box = gt[..., 1:]

        return objectness_score, proposals, gt_bounding_box

    def forward(
        self,
        objectness_score: torch.Tensor,
        proposals: torch.Tensor,
        gt_bounding_box: torch.Tensor,
    ) -> torch.Tensor:
        """
        https://arxiv.org/pdf/1506.01497
        Calculates IoU between ground truth and predicted bounding boxes.
        Boxes with IoU > positive_threshold are positive ones and boxes
        with IoU < negative_threshold are negative. Others are not considered.
        We take n_cls / 2 positive boxes (if there are that many) and the rest are negative.

        Args:
            objectness_score: Objectness score for bounding boxes (num_proposals, 4).
            proposals: Predicted bounding boxes by the RPN of shape (num_proposals, 4). # TODO: make it work for N > 1
            gt_bounding_box: Ground truth object bounding boxes (num_objects_in_frame, 4).

        Returns:
            Binary Cross Entropy Loss between positive and negative boxes.
        """
        iou_matrix = box_iou(boxes1=gt_bounding_box.squeeze(0), boxes2=proposals)

        # For each gt object first find best predicted anchor (with highest IoU) and all that satisfy the threshold condition
        num_objects, num_proposals = iou_matrix.shape
        best_iou_scores, flat_indices = torch.topk(iou_matrix.flatten(), k=num_objects)
        best_pos_rows = flat_indices // num_proposals
        best_pos_cols = flat_indices % num_proposals

        pos_rows, pos_cols = torch.where(iou_matrix > self.iou_positive_threshold)
        neg_rows, neg_cols = torch.where(iou_matrix < self.iou_negative_threshold)

        # # If best predicted anchor IoU < iou_negative_threshold we remove it from best_pos_cols
        # keep = ~(best_iou_scores < self.iou_negative_threshold)
        # best_pos_cols = best_pos_cols[keep]
        # best_pos_rows = best_pos_rows[keep]

        pos_cols = torch.cat([best_pos_cols, pos_cols])
        pos_rows = torch.cat([best_pos_rows, pos_rows])

        num_pos = min(self.n_cls // 2, pos_cols.numel())
        num_neg = min(self.n_cls - num_pos, neg_cols.numel())

        pos_indices = torch.randperm(n=pos_cols.shape[0])
        pos_cols = pos_cols[pos_indices][:num_pos]
        pos_rows = pos_rows[pos_indices]

        neg_indices = torch.randperm(n=neg_cols.shape[0])
        neg_cols = neg_cols[neg_indices][:num_neg]
        neg_rows = neg_rows[neg_indices]

        pos = objectness_score[pos_cols]
        neg = objectness_score[neg_cols]

        pred_classification = torch.cat([pos, neg])
        gt_classification = torch.cat([torch.ones(num_pos), torch.zeros(num_neg)]).to(
            pred_classification.device
        )

        gt_bounding_box = gt_bounding_box[:, pos_rows, :].squeeze(0)
        pred_bounding_box = proposals[pos_cols, :]

        return self.classification_loss(
            pred_classification, gt_classification
        ) + self.regularization_factor * self.regression_loss(
            pred_bounding_box, gt_bounding_box
        )


class RCNNCrossEntropyAndRegressionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.iou_threshold = 0.3
        self.n_cls = 256
        self.classification_loss = nn.CrossEntropyLoss(reduction="mean")
        self.regression_loss = nn.SmoothL1Loss(reduction="mean")
        self.register_forward_pre_hook(self._extract_relevant_tensor_info)

    @staticmethod
    def _extract_relevant_tensor_info(
        module: nn.Module,
        inputs: tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred_info, gt = inputs
        pred_class_logits, pred_bounding_boxes = pred_info["faster-rcnn"]
        gt_class, gt_bounding_boxes = gt[..., 0], gt[..., 1:]

        return pred_class_logits, pred_bounding_boxes, gt_class, gt_bounding_boxes

    def forward(
        self,
        pred_class_logits: torch.Tensor,
        pred_bounding_boxes: torch.Tensor,
        gt_class: torch.Tensor,
        gt_bounding_boxes: torch.Tensor,
    ) -> torch.Tensor:
        iou_matrix = box_iou(
            boxes1=gt_bounding_boxes.squeeze(0), boxes2=pred_bounding_boxes
        )

        # object_rows represents the gt object corresponding to the predicted object
        # which is represented by the object_cols in the iou_matrix if IoU > threshold
        gt_indices, pred_indices = torch.where(iou_matrix > self.iou_threshold)

        if gt_indices.shape[0] == 0:
            # no matches in early training, RPN is bad, so we match
            max_ious, pred_indices = iou_matrix.max(dim=1)
            keep = torch.where(max_ious > 0, True, False)
            max_ious, pred_indices = max_ious[keep], pred_indices[keep]
            gt_indices = torch.where(keep == True)[0]

        # filter boxes if they satisfy IoU condition
        pred_class_logits = pred_class_logits[pred_indices]
        pred_bounding_boxes = pred_bounding_boxes[pred_indices]

        gt_classes_per_pred = gt_class[:, gt_indices].squeeze(0)
        gt_bounding_boxes_per_pred = gt_bounding_boxes[:, gt_indices, :].squeeze(0)

        return self.classification_loss(
            pred_class_logits, gt_classes_per_pred.to(torch.int64)
        ) + self.regression_loss(pred_bounding_boxes, gt_bounding_boxes_per_pred)
