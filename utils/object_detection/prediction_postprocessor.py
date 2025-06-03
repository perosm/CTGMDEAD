import torch
from torch import nn
from utils.object_detection_3d.utils import (
    inverse_normalize_predicted_keypoints_by_proposals,
    project_2d_points_to_3d_points,
)


class PredictionPostprocessor(nn.Module):
    def __init__(self):
        super().__init__()

    def _fetch_object_detection_3d_info(
        self,
        bounding_boxes_2d: torch.Tensor,
        distance_head_output: torch.Tensor,
        size: torch.Tensor,
        yaw: torch.Tensor,
        keypoints: torch.Tensor,
        projection_matrix: torch.Tensor,
    ) -> torch.Tensor:
        KEYPOINT_INDEX = -1
        projection_matrix = projection_matrix.squeeze(0)
        focal_x = projection_matrix[0, 0]
        num_objects = bounding_boxes_2d.shape[0]

        # Fetch distance head information
        H, _, h_rec, _ = distance_head_output.unbind(dim=-1)
        distance = focal_x * H * h_rec

        # Fetch rotation angle
        sin_theta, cos_theta = yaw.unbind(dim=-1)
        rotation_y = torch.atan2(sin_theta, cos_theta).view(num_objects, -1)

        # Fetch object center in 3D space
        inverse_normalized_keypoints = (
            inverse_normalize_predicted_keypoints_by_proposals(
                pred_keypoints=keypoints, pos_proposals=bounding_boxes_2d
            )
        )
        object_center_2d = inverse_normalized_keypoints[:, :, KEYPOINT_INDEX]
        object_center_3d = project_2d_points_to_3d_points(
            points_2d=object_center_2d,
            depth=distance,
            projection_matrix=projection_matrix,
        )
        return torch.cat([size, object_center_3d, rotation_y], dim=-1)

    def forward(
        self,
        prediction: dict[str, tuple[torch.Tensor, ...]],
        projection_matrix: torch.Tensor,
    ) -> dict[str, tuple[torch.Tensor, ...]]:
        filtered_proposals = prediction["rpn"]
        class_probits, bounding_boxes_2d, labels = prediction["faster-rcnn"]
        distance_head_output, size, yaw, keypoints = prediction["mono-rcnn"]

        bounding_boxes_3d = self._fetch_object_detection_3d_info(
            bounding_boxes_2d=bounding_boxes_2d,
            distance_head_output=distance_head_output,
            size=size,
            yaw=yaw,
            keypoints=keypoints,
            projection_matrix=projection_matrix,
        )

        return torch.cat(
            [
                labels.unsqueeze(-1),
                class_probits.unsqueeze(-1),
                bounding_boxes_2d,
                bounding_boxes_3d,
            ],
            dim=-1,
        )
