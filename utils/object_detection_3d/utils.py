import torch


def project_3d_boxes_to_image(
    boxes_3d: torch.Tensor, projection_matrix: torch.Tensor
) -> torch.Tensor:
    """
    Args:
        boxes_3d: Bounding boxes in 3D space of shape (num_boxes, 7).
        projection_matrix: Projection matrix that projects 3D bounding boxes onto image (4, 4).

    Returns:
        Projected bounding boxes.
    """
    num_objects = boxes_3d.shape[0]
    h, w, l = boxes_3d[:, 0:3].T
    x, y, z = boxes_3d[:, 3:6].T
    rotation_y = boxes_3d[:, 6]
    zeros = torch.zeros(num_objects, device=boxes_3d.device)
    ones = torch.ones(num_objects, device=boxes_3d.device)
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

    ones = torch.ones((num_objects, 1, 8), device=boxes_3d.device)
    corners_homogeneous = torch.cat([corners, ones], dim=1)

    projections = projection_matrix.unsqueeze(0) @ corners_homogeneous
    projections = projections[:, :2, :] / projections[:, 2, :].unsqueeze(1)

    return projections.to(torch.int16)
