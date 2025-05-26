import torch
from torchvision.ops import box_iou


def project_3d_boxes_to_image(
    boxes_3d_info: torch.Tensor, projection_matrix: torch.Tensor
) -> torch.Tensor:
    """
    Extracts the center of the object in camera coordinate system and it's height, width and length,
    Calculates the eigth vertices of the 3D bounding box and projects it onto the image.

       (5)-----(4)
       /|       /|
      / |      / |
    (6)-----(7)  |
     |  |    |   |
     | (1)---|--(0)
     | /     |  /
     |/      | /
    (2)-----(3)
    Args:
        boxes_3d_info: Bounding boxes in 3D space of shape (num_boxes, 8).
        projection_matrix: Projection matrix of shape (3, 4) that projects 3D bounding boxes onto image.

    Returns:
        Projected bounding boxes.
    """
    num_objects = boxes_3d_info.shape[0]
    h, w, l = boxes_3d_info[:, 0:3].T
    x, y, z = boxes_3d_info[:, 3:6].T
    rotation_y = boxes_3d_info[:, 6]
    zeros = torch.zeros(num_objects, device=boxes_3d_info.device)
    ones = torch.ones(num_objects, device=boxes_3d_info.device)
    R = torch.stack(
        [
            torch.stack([torch.cos(rotation_y), zeros, torch.sin(rotation_y)], dim=0),
            torch.stack([zeros, ones, zeros], dim=0),
            torch.stack([-torch.sin(rotation_y), zeros, torch.cos(rotation_y)], dim=0),
        ],
        dim=0,
    ).permute(2, 0, 1)
    x_corners = torch.stack(
        [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2], dim=1
    )
    y_corners = torch.stack([zeros, zeros, zeros, zeros, -h, -h, -h, -h], dim=1)
    z_corners = torch.stack(
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], dim=1
    )
    corners = torch.stack([x_corners, y_corners, z_corners], dim=1)
    corners = torch.einsum("nij,njk->nik", R, corners)
    corners[:, 0] += x.unsqueeze(1)
    corners[:, 1] += y.unsqueeze(1)
    corners[:, 2] += z.unsqueeze(1)

    ones = torch.ones((num_objects, 1, 8), device=boxes_3d_info.device)
    corners_homogeneous = torch.cat([corners, ones], dim=1)

    projections = projection_matrix.unsqueeze(0) @ corners_homogeneous
    projections = projections[:, :2, :] / projections[:, 2, :].unsqueeze(1)

    return projections.to(torch.int16)


def project_3d_points_to_image(
    points_3d: torch.Tensor, projection_matrix: torch.Tensor
) -> torch.Tensor:
    """
    Projects 3d point from world to pixel coordinates.

    Args:
        point_3d: Points to be projected from camera coordinates to pixel coordinates of shape (num_points, 3).
        projection_matrix: Camera projection matrix used of shape (3, 4).

    Returns:
        Point 2D projected to pixel coordinates of shape (num_points, 2).
    """
    num_points = points_3d.shape[0]
    ones = torch.ones(num_points, 1).to(device=points_3d.device)
    homogeneous_coordinates_3d = torch.cat([points_3d, ones], dim=1)
    homogeneous_coordinates_2d = (projection_matrix @ homogeneous_coordinates_3d.T).T

    return (homogeneous_coordinates_2d / homogeneous_coordinates_2d[:, 2][:, None])[
        :, :2
    ].to(torch.int16)


def match_proposals_to_objects(
    proposals: torch.Tensor, gt_box_2d: torch.Tensor, iou_threshold: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Matches proposals to ground truth bounding boxes if their IoU > iou_positive_threshold.

    Args:
        proposals: Proposals output by the RPN of shape (num_proposals, 4).
        gt_box_2d: Ground truth 2D bounding boxes of shape (num_objects, 4).

    Returns:
        Indices that pair proposals and ground truth.
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
    positive_mask = max_iou_per_proposal > iou_threshold
    positive_mask[gt_best_proposal_indices] = True  # TODO: remove comment?

    pos_proposal_indices = torch.where(positive_mask)[0]
    pos_gt_indices = gt_indices[pos_proposal_indices]

    return pos_proposal_indices, pos_gt_indices


def normalize_keypoints_by_proposals(
    keypoints: torch.Tensor, pos_proposals: torch.Tensor
) -> torch.Tensor:
    """
    Used when training.
    Let (x1, y1, x2, y2) denote top-left and bottom-right corners of the proposals
    and (p1, p2) denote the keypoint coordinates, the keypoints are normalized as follows:

    t1 = (p1 - x1) / (x2 - x1)
    t2 = (p2 - y1) / (y2 - y1)

    Args:
        keypoints: Projected bounding box 3d keypoints of shape (num_pos_proposals, 2, 9).
        proposals: Positive proposals of shape (num_pos_proposals, 4).

    Returns:
        Keypoints normalized by the above formula of shape (num_pos_proposals, 2, 9).
    """
    num_pos_proposals = pos_proposals.shape[0]
    x1, y1, x2, y2 = pos_proposals.unbind(dim=-1)
    keypoints[:, 0, :] = (keypoints[:, 0, :] - x1.view(num_pos_proposals, -1)) / (
        x2.view(num_pos_proposals, -1) - x1.view(num_pos_proposals, -1)
    )
    keypoints[:, 1, :] = (keypoints[:, 1, :] - y1.view(num_pos_proposals, -1)) / (
        y2.view(num_pos_proposals, -1) - y1.view(num_pos_proposals, -1)
    )

    return keypoints


def inverse_normalize_keypoints_by_proposals(
    keypoints: torch.Tensor, pos_proposals: torch.Tensor
) -> torch.Tensor:
    """
    Used when evaluating.
    Let (x1, y1, x2, y2) denote top-left and bottom-right corners of the proposals
    and (p1, p2) denote the keypoint coordinates, the keypoints are unnormalized as follows:

    p1 = x1 + t1 * (x2 - x1)
    p2 = y1 + t2 * (y2 - y1)

    Args:
        keypoints: Projected and normalized bounding box 3d keypoints of shape (num_pos_proposals, 18).
        proposals: Positive proposals of shape (num_pos_proposals, 4).

    Returns:
        Unnormalized keypoints by the above formula of shape (num_pos_proposals, 18).
    """
    num_pos_proposals = pos_proposals.shape[0]
    x1, y1, x2, y2 = pos_proposals.unbind(dim=-1)
    keypoints[:, 0, :] = x1.view(num_pos_proposals, -1) + keypoints[:, 0, :] * (
        x2.view(num_pos_proposals, -1) - x1.view(num_pos_proposals, -1)
    )
    keypoints[:, 1, :] = y1.view(num_pos_proposals, -1) + keypoints[:, 1, :] * (
        y2.view(num_pos_proposals, -1) - y1.view(num_pos_proposals, -1)
    )

    return keypoints
