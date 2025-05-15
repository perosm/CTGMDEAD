import torch


def apply_deltas_to_boxes(boxes: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """
    Applies parametrized offsets/deltas from the RPN head.

    First it fetches upper left and lower right corner of a box,
    then the output of the RPN head is fetched in dx, dy, dw, dh format.
    Transformation is applied onto parametrized output to obtain output
    in pixels. The output in pixels is then applied onto the boxes.

    Args:
        - boxes: Boxes of shape (num_boxes, 4).
        - deltas: Output of the RPN heads of shape (num_deltas, 4).

    Returns:
        - transformed_boxes: Boxes on which deltas are applied.
    """
    # corners
    x1, y1, x2, y2 = boxes.unbind(dim=-1)  # left, top, right, bottom
    box_width = x2 - x1
    box_height = y2 - y1
    box_center_x = x1 + box_width / 2
    box_center_y = y1 + box_height / 2

    deltas_x, deltas_y, deltas_w, deltas_h = deltas.unbind(dim=-1)

    # center coordinates and height, width
    center_x = box_center_x + deltas_x * box_width
    center_y = box_center_y + deltas_y * box_height
    width = box_width * torch.exp(deltas_w)
    height = box_height * torch.exp(deltas_h)

    # switching back to corners
    x1 = center_x - width / 2
    y1 = center_y - height / 2
    x2 = center_x + width / 2
    y2 = center_y + height / 2

    transformed_boxes = torch.stack([x1, y1, x2, y2], dim=-1)

    return transformed_boxes


def get_deltas_from_bounding_boxes(
    reference_boxes: torch.Tensor, predicted_boxes: torch.Tensor
) -> torch.Tensor:
    x1, y1, x2, y2 = reference_boxes.unbind(dim=-1)
    reference_boxes_width = x2 - x1
    reference_boxes_height = y2 - y1
    reference_boxes_center_x = x1 + reference_boxes_width / 2
    reference_boxes_center_y = y1 + reference_boxes_height / 2

    predicted_boxes_x1, predicted_boxes_y1, predicted_boxes_x2, predicted_boxes_y2 = (
        predicted_boxes.unbind(dim=-1)
    )
    predicted_boxes_width = predicted_boxes_x2 - predicted_boxes_x1
    predicted_boxes_height = predicted_boxes_y2 - predicted_boxes_y1
    predicted_boxes_center_x = predicted_boxes_x1 + predicted_boxes_width / 2
    predicted_boxes_center_y = predicted_boxes_y1 + predicted_boxes_height / 2

    deltas_x = (
        reference_boxes_center_x - predicted_boxes_center_x
    ) / predicted_boxes_width
    deltas_y = (
        reference_boxes_center_y - predicted_boxes_center_y
    ) / predicted_boxes_height
    deltas_w = torch.log(reference_boxes_width / predicted_boxes_width)
    deltas_h = torch.log(reference_boxes_height / predicted_boxes_height)

    deltas = torch.stack([deltas_x, deltas_y, deltas_w, deltas_h], dim=1)

    return deltas
