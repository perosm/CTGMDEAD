import torch
from torch import nn
from torchvision.ops import clip_boxes_to_image, nms


class OutputHeads(nn.Module):
    rpn_out_channels = 256
    output_features = 256

    def __init__(
        self,
        training: bool,
        image_size: tuple[int, int],
        pool_size: list[int, int],
        num_classes: int,
        score_threshold: float,
        iou_threshold: float,
        num_detections: int,
        top_k_boxes_testing: int,
    ) -> None:
        """
        Regression and classification heads for the Faster RCNN module.

        Args:
            pool_size: Height and width of the feature maps after applying RoI align.
            num_classes: Number of classes we are trying to predict + 1 for background.
        """
        super().__init__()
        self.training = training
        self.image_size = image_size
        self.pool_size = pool_size
        self.num_classes = num_classes
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.num_detection = num_detections
        self.top_k_boxes_testing = top_k_boxes_testing
        self.fc1 = nn.Linear(
            in_features=self.rpn_out_channels * pool_size[0] * pool_size[1],
            out_features=self.output_features,
        )
        self.classification_head = nn.Linear(
            in_features=self.output_features, out_features=num_classes
        )
        self.regression_head = nn.Linear(
            in_features=self.output_features, out_features=4 * num_classes
        )

    def _fetch_boxes(
        self,
        proposals: torch.Tensor,
        bbox_regression_deltas: torch.Tensor,
        class_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fetches proposals given anchors and bbox_regression_deltas for each of the anchors.

        Args:
            - proposals:
            - bbox_regression_deltas:

        Returns:
            - bounding_boxes:
        """
        # corners of the anchors
        x1, y1, x2, y2 = proposals.unbind(dim=-1)  # left, top, right, bottom
        proposal_width = x2 - x1
        proposal_height = y2 - y1
        proposal_center_x = y1 + proposal_width / 2
        proposal_center_y = x1 + proposal_height / 2

        labels = torch.argmax(class_logits, dim=-1)
        bbox_regression_deltas = bbox_regression_deltas.view(
            -1, class_logits.shape[-1], 4
        )
        labels = labels.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 4)
        bbox_regression_deltas = torch.gather(
            bbox_regression_deltas, 1, labels
        ).squeeze(1)
        deltas_x, deltas_y, deltas_w, deltas_h = bbox_regression_deltas.unbind(dim=-1)

        # center coordinates and height, width of predicted bounding box
        pred_x = proposal_center_x + deltas_x * proposal_width
        pred_y = proposal_center_y + deltas_y * proposal_height
        pred_width = proposal_width * torch.exp(deltas_w)
        pred_height = proposal_height * torch.exp(deltas_h)

        # switching back to corners of predicted bounding box
        pred_x1 = pred_x - pred_width / 2
        pred_y1 = pred_y - pred_height / 2
        pred_x2 = pred_x + pred_width / 2
        pred_y2 = pred_y + pred_height / 2

        bounding_boxes = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=-1)

        return bounding_boxes

    def _filter_detections(
        self,
        bounding_boxes: torch.Tensor,
        class_logits: torch.Tensor,
    ):
        """
        Detections are being filtered in the following order:
            - 1) Clip to fit into image
            - 2) Boxes with class probability < score_threshold are removed
            - 3) Pre Non-Max Supression (NMS) filtering
            - 4) Non-Max Supression (NMS) between proposals for the same ground truth
            - 5) Pick top K proposals

        Args:
            bounding_boxes: Bounding boxes .
            class_logits: Class logits

        Returns:
            class_probits:

        """
        # TODO: Make it work for N > 1
        # 1) clip bounding boxes to fit into image
        bounding_boxes = clip_boxes_to_image(boxes=bounding_boxes, size=self.image_size)

        # 2) filter proposals with class probability < score_threshold are remove
        class_probits = torch.softmax(class_logits, dim=-1)
        keep = torch.any(class_probits > self.score_threshold, dim=1)
        bounding_boxes = bounding_boxes[keep]
        class_probits = class_probits[keep]

        # 3) filter using NMS
        indices = torch.argmax(class_probits, dim=1)
        highest_class_probits = torch.gather(
            class_probits, 1, indices.unsqueeze(-1)
        ).squeeze(-1)
        keep = nms(
            boxes=bounding_boxes,
            scores=highest_class_probits,
            iou_threshold=self.iou_threshold,
        )
        class_probits = class_probits[keep]
        bounding_boxes = bounding_boxes[keep]

        # 4) pick top K proposals
        if not self.training:
            keep = keep[: self.top_k_boxes_testing]

            bounding_boxes = bounding_boxes[keep]
            objectness_score = objectness_score[keep]

        return class_probits, bounding_boxes

    def forward(
        self, pooled_proposals_per_feature_map: torch.Tensor, proposals: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        num_proposals = pooled_proposals_per_feature_map.shape[0]
        intermediary = self.fc1(
            pooled_proposals_per_feature_map.view(num_proposals, -1)
        )
        class_logits = self.classification_head(intermediary)
        bounding_box_deltas = self.regression_head(intermediary)

        bounding_boxes = self._fetch_boxes(
            proposals=proposals,
            bbox_regression_deltas=bounding_box_deltas,
            class_logits=class_logits,
        )
        class_probits, bounding_boxes = self._filter_detections(
            bounding_boxes=bounding_boxes, class_logits=class_logits
        )
        return class_probits, bounding_boxes
