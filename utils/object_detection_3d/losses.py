import torch
from torch import nn
import torch.nn.functional as F

from utils.shared.enums import ObjectDetectionEnum
from utils.object_detection_3d.utils import (
    match_proposals_to_objects,
    project_3d_boxes_to_image,
    project_3d_points_to_image,
    normalize_gt_keypoints_by_proposals,
)

PROPOSAL_INDEX = 4
GT_BOX_2D_SLICE = slice(
    ObjectDetectionEnum.box_2d_left, ObjectDetectionEnum.box_2d_bottom + 1
)


class UncertaintyAwareRegressionLoss(nn.Module):
    def __init__(self, name: str = "UncertaintyAwareRegressionLoss"):
        super().__init__()
        self.name = name
        self.lambda_H = 1
        self.lambda_h_rec = 1
        self.iou_positive_threshold = 0.5
        self.register_forward_pre_hook(self._extract_relevant_tensor_info)

    @staticmethod
    def _extract_relevant_tensor_info(
        module: nn.Module,
        inputs: tuple[dict[str, tuple], dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        DISTANCE_HEAD_INDEX = 0
        prediction, ground_truth = inputs

        # Prepare predictions and groundtruth
        distance_head_output = prediction["mono-rcnn"][DISTANCE_HEAD_INDEX]
        gt_object_info = ground_truth["gt_info"].squeeze(0)

        # Match predictions and grountruth
        pos_proposal_indices, pos_gt_indices = match_proposals_to_objects(
            proposals=prediction["rpn"][PROPOSAL_INDEX],
            gt_box_2d=gt_object_info[..., GT_BOX_2D_SLICE],
        )

        # Extract distance head info
        pred_H, pred_log_sigma_H, pred_h_rec, pred_log_sigma_h_rec = torch.unbind(
            distance_head_output[pos_proposal_indices], dim=-1
        )

        # Extract ground truth info
        pos_gt_object_info = gt_object_info[pos_gt_indices]
        projection_matrix = ground_truth["projection_matrix"].squeeze(0)

        gt_H = pos_gt_object_info[:, ObjectDetectionEnum.height]
        gt_distance = pos_gt_object_info[:, ObjectDetectionEnum.z]
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
        L_H = F.l1_loss(pred_H, gt_H, reduction="none")
        # / torch.exp(pred_log_sigma_H)
        #     + self.lambda_H * pred_log_sigma_H
        # )
        L_h_rec = F.l1_loss(pred_h_rec, gt_h_rec, reduction="none")
        #     / torch.exp(pred_log_sigma_h_rec)
        #     + self.lambda_h_rec * pred_log_sigma_h_rec
        # )

        return (L_H + L_h_rec).mean()


class L1SizeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda_size = 3
        self.register_forward_pre_hook(self._extract_relevant_tensor_info)

    @staticmethod
    def _extract_relevant_tensor_info(
        module: nn.Module,
        inputs: tuple[dict[str, tuple], dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        PRED_SIZE_INDEX = 1
        GT_SIZE_SLICE = slice(
            ObjectDetectionEnum.height, ObjectDetectionEnum.length + 1
        )
        prediction, ground_truth = inputs

        # Match predictions and ground truth
        gt_object_info = ground_truth["gt_info"].squeeze(0)
        pos_proposal_indices, pos_gt_indices = match_proposals_to_objects(
            proposals=prediction["rpn"][PROPOSAL_INDEX],
            gt_box_2d=gt_object_info[:, GT_BOX_2D_SLICE],
        )
        pred_size = prediction["mono-rcnn"][PRED_SIZE_INDEX]
        gt_size = gt_object_info[:, GT_SIZE_SLICE]

        return pred_size[pos_proposal_indices], gt_size[pos_gt_indices]

    def forward(self, pred_size: torch.Tensor, gt_size: torch.Tensor) -> torch.Tensor:
        return self.lambda_size * F.l1_loss(pred_size, gt_size, reduction="mean")


class L1YawLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_forward_pre_hook(self._extract_relevant_tensor_info)
        self.lambda_yaw = 5

    @staticmethod
    def _extract_relevant_tensor_info(
        module: nn.Module,
        inputs: tuple[dict[str, tuple], dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        PRED_YAW_INDEX = 2
        prediction, ground_truth = inputs

        # Prepare predictions and ground truth
        gt_object_info = ground_truth["gt_info"].squeeze(0)
        pos_proposal_indices, pos_gt_indices = match_proposals_to_objects(
            proposals=prediction["rpn"][PROPOSAL_INDEX],
            gt_box_2d=gt_object_info[:, GT_BOX_2D_SLICE],
        )
        pred_yaw = prediction["mono-rcnn"][PRED_YAW_INDEX]
        gt_yaw = gt_object_info[:, ObjectDetectionEnum.rotation_y]

        # Convert yaw angle to same space as predictions
        gt_yaw = torch.stack([torch.sin(gt_yaw), torch.cos(gt_yaw)], dim=-1)

        return pred_yaw[pos_proposal_indices], gt_yaw[pos_gt_indices]

    def forward(self, pred_yaw: torch.Tensor, gt_yaw: torch.Tensor) -> torch.Tensor:
        return self.lambda_yaw * F.l1_loss(pred_yaw, gt_yaw, reduction="mean")


class L1KeypointsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda_keypoints = 5
        self.register_forward_pre_hook(self._extract_relevant_tensor_info)

    @staticmethod
    def _extract_relevant_tensor_info(
        module: nn.Module,
        inputs: tuple[dict[str, tuple], dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        PRED_KEYPOINTS_INDEX = 3
        GT_BOX_3D_CENTER_SLICE = slice(ObjectDetectionEnum.x, ObjectDetectionEnum.z + 1)
        GT_BOX_3D_SLICE = slice(
            ObjectDetectionEnum.height, ObjectDetectionEnum.rotation_y + 1
        )
        prediction, ground_truth = inputs

        # fetch relevant info
        gt_box_info = ground_truth["gt_info"].squeeze(0)
        proposals = prediction["rpn"][PROPOSAL_INDEX]

        # Match proposals to ground truth
        pos_proposal_indices, pos_gt_indices = match_proposals_to_objects(
            proposals=proposals,
            gt_box_2d=gt_box_info[:, GT_BOX_2D_SLICE],
        )

        # Extract relevant object center, 3d object info, projection matrix
        gt_box_3d_center = gt_box_info[:, GT_BOX_3D_CENTER_SLICE]
        gt_box_3d_info = gt_box_info[:, GT_BOX_3D_SLICE]
        projection_matrix = ground_truth["projection_matrix"].squeeze(0)

        # Project 3d ground truth to image
        gt_box_3d_center_projected = project_3d_points_to_image(
            points_3d=gt_box_3d_center, projection_matrix=projection_matrix
        )
        gt_box_3d_projected = project_3d_boxes_to_image(
            boxes_3d_info=gt_box_3d_info, projection_matrix=projection_matrix
        )

        num_objects = gt_box_info.shape[0]
        pred_keypoints = prediction["mono-rcnn"][PRED_KEYPOINTS_INDEX]
        gt_keypoints = torch.cat(
            [
                gt_box_3d_projected.view(num_objects, 2, 8),
                gt_box_3d_center_projected.view(num_objects, 2, -1),
            ],
            dim=-1,
        )
        # We only use proposals that are positive (i.e. overlap with g.t. > threshold)
        pos_proposals = proposals[pos_proposal_indices]
        pos_pred_keypoints = pred_keypoints[pos_proposal_indices]
        pos_gt_keypoints = gt_keypoints[pos_gt_indices]

        # Normalize ground truth positives
        gt_normalized_keypoints = normalize_gt_keypoints_by_proposals(
            pos_gt_keypoints=pos_gt_keypoints, pos_proposals=pos_proposals
        )

        return pos_pred_keypoints.flatten(1), gt_normalized_keypoints.flatten(1)

    def forward(
        self,
        pred_normalized_keypoints: torch.Tensor,
        gt_normalized_keypoints: torch.Tensor,
    ) -> torch.Tensor:
        return F.l1_loss(
            pred_normalized_keypoints, gt_normalized_keypoints, reduction="mean"
        )
