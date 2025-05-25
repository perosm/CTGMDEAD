import torch
from torch import nn
import torch.nn.functional as F

from utils.object_detection_3d.utils import project_3d_boxes_to_image
from torchvision.ops import box_iou
from utils.shared.enums import ObjectDetectionEnum


class UncertaintyAwareRegressionLoss(nn.Module):
    def __init__(self, name: str = "UncertaintyAwareRegressionLoss"):
        super().__init__()
        self.name = name
        self.lambda_H = 0.25
        self.lambda_h_rec = 1
        self.iou_positive_threshold = 0.5
        self.register_forward_pre_hook(self._extract_relevant_tensor_info)

    def _extract_relevant_tensor_info(
        self,
        module: nn.Module,
        inputs: tuple[dict[str, tuple], dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        prediction, ground_truth = inputs

        proposals_index = 1

        # prepare predictions
        proposals = prediction["faster-rcnn"][proposals_index]
        gt_info = ground_truth["gt_info"]
        gt_box_2d = gt_info[
            ...,
            ObjectDetectionEnum.box_2d_left : ObjectDetectionEnum.box_2d_bottom + 1,
        ].squeeze(0)
        pos_proposal_indices, pos_gt_indices = self._match_proposals_to_objects(
            proposals=proposals, gt_box_2d=gt_box_2d
        )
        pos_proposals = proposals[pos_proposal_indices]

        pred_H, pred_log_sigma_H, pred_h_rec, pred_log_sigma_h_rec = torch.unbind(
            prediction["distance_head"][pos_proposal_indices], dim=-1
        )

        # prepare ground truth
        gt_object_info = ground_truth["gt_info"].squeeze(0)
        gt_object_info = gt_object_info[pos_gt_indices]
        projection_matrix = ground_truth["projection_matrix"].squeeze(0)
        gt_H = gt_object_info[:, ObjectDetectionEnum.height]
        gt_distance = gt_object_info[:, ObjectDetectionEnum.z]
        focal_x = projection_matrix[0, 0]
        gt_h_rec = gt_distance / (focal_x * gt_H)

        return (
            pred_H,
            pred_log_sigma_H,
            gt_H,
            pred_h_rec,
            pred_log_sigma_h_rec,
            gt_h_rec,
        )

    def _match_proposals_to_objects(
        self, proposals: torch.Tensor, gt_box_2d: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Matches proposals to ground truth bounding boxes if their IoU > iou_positive_threshold.
        Args:
            proposals:
            gt_box_2d:

        Returns:
            Indices that pair proposals and ground truth
        """
        # Computing IoU between gt and all anchors
        iou_matrix = box_iou(boxes1=gt_box_2d, boxes2=proposals)

        # For each proposal find gt with highest IoU
        max_iou_per_proposal, gt_indices = iou_matrix.max(dim=0)

        # For each gt find anchor with highest IoU
        gt_best_proposal_indices = iou_matrix.argmax(dim=1)

        # For positive proposals we take all anchors whose
        # IoU > iou_positive_threshold
        # and we force always at least one anchor per gt
        positive_mask = max_iou_per_proposal > self.iou_positive_threshold
        positive_mask[gt_best_proposal_indices] = True  # TODO: remove comment?

        pos_proposal_indices = torch.where(positive_mask)[0]
        pos_gt_indices = gt_indices[pos_proposal_indices]

        return pos_proposal_indices, pos_gt_indices

    def forward(
        self,
        pred_H: torch.Tensor,
        pred_log_sigma_H: torch.Tensor,
        gt_H: torch.Tensor,
        pred_h_rec: torch.Tensor,
        pred_log_sigma_h_rec: torch.Tensor,
        gt_h_rec: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculates uncertainty aware loss from the paper: https://arxiv.org/pdf/2104.03775.

        Args:
            pred_H: Predicted objects height in meters of shape (num_proposals).
            pred_log_sigma_H: Log uncertainty of predicted height of shape (num_proposals).
            gt_H:
            pred_h_rec:
            pred_log_sigma_h_rec:
            gt_h_rec:
        Returns:
            Uncertainty aware regression loss for height in world coordinate frame
            and reciprocal height in camera coordinate frame.
        """
        L_H = (
            F.l1_loss(pred_H, gt_H, reduction="none") / torch.exp(pred_log_sigma_H)
            + self.lambda_H * pred_log_sigma_H
        )
        L_h_rec = (
            F.l1_loss(pred_h_rec, gt_h_rec, reduction="none")
            / torch.exp(pred_log_sigma_h_rec)
            + self.lambda_h_rec * pred_log_sigma_h_rec
        )

        return (L_H + L_h_rec).mean()
