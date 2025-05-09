import torch
from torch import nn
from torchvision.ops import box_iou


class RPNClassificationLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.n_cls = 256
        self.iou_threshold = 0.5
        self.loss = nn.BCELoss(reduction="mean")
        self.register_forward_pre_hook(self._extract_relevant_tensor_info)

    @staticmethod
    def _extract_relevant_tensor_info(
        module: nn.Module,
        inputs: tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred_info, y_pred = inputs
        objectness_score, proposals = pred_info["rpn"]

        return objectness_score, proposals, y_pred

    def forward(
        self,
        objectness_score: torch.Tensor,
        proposals: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculates IoU between ground truth and predicted bounding boxes.
        Boxes with IoU > threshold are positive ones and others are negative.
        We take n_cls / 2 positive boxes and the rest are negative.

        Args:
            y_pred: Object bounding boxes of shape (num_proposals, 4).
            y_true: Predicted object bounding boxes (num_proposals, 4).

        Returns:
            Binary Cross Entropy Loss between positive and negative boxes.
        """
        iou_matrix = box_iou(boxes1=y_true, boxes2=proposals)
        mask = torch.where(iou_matrix > self.iou_threshold, 1, 0)
        # return self.loss(y_pred, y_true)


class RPNRegressionLoss(nn.Module):
    def __init__(self, regularization_factor: float = 10.0) -> None:
        super().__init__()
        self.regularization_factor = regularization_factor
        self.loss = nn.SmoothL1Loss(reduction="mean")
        self.register_forward_pre_hook(self._extract_relevant_tensor_info)

    @staticmethod
    def _extract_relevant_tensor_info(
        module: nn.Module,
        inputs: tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred_info, y_pred = inputs
        objectness_score, proposals = pred_info["rpn"]

        return objectness_score, proposals, y_pred

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.regularization_factor * self.loss(y_pred, y_true)


class RCNNCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.loss(y_pred, y_true)


if __name__ == "__main__":
    rpn_classification_loss = RPNClassificationLoss()
    rpn_classification_loss(
        {
            "rpn": (torch.ones(10), torch.ones(100)),
            "faster-rcnn": (torch.ones(10), torch.ones(10)),
        },
        torch.ones(10),
    )
