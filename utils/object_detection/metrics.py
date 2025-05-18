import torch
from torch import nn

from torchvision.ops import box_iou


class mAP(nn.Module):
    def __init__(self):
        """
        COCO style mAP.
        """
        self.num_classes = 3
        super().__init__()
        self.eval()

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Although we predict num_classes + background (class 0), we filter it before,
        and we only take into consideration classes from 1 to num_classes.

        Args:
            - pred: Predicted objects of format (label, top, left, bottom, right).
            - gt: Ground truth object of format (label, top, left, bottom, right).
        """
        iou_matrix = box_iou(boxes1=gt, boxes2=pred)
        thresholds = torch.arange(start=0.5, end=1.0, step=0.05)
        thresholds_classes_matrix = torch.empty(
            size=(thresholds.shape[0], self.num_classes)
        )

        for i in range(thresholds.shape[0]):
            for c in range(1, self.num_classes + 1):
                pass
