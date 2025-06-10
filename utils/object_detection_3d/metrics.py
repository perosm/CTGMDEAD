import torch
from torch import nn

from utils.object_detection.metrics import mAP_pascal_voc
from utils.shared.enums import ObjectDetectionEnum
from utils.object_detection_3d.utils import project_3d_boxes_to_bev


GT_BOUNDING_BOX_3D_SLICE = slice(ObjectDetectionEnum.height, None)
PRED_BOUNDING_BOX_3D_SLICE = slice(6, None)


class mAP_BEV(nn.Module):
    higher = True

    def __init__(self):
        super().__init__()
        self.eval()
        self.register_forward_pre_hook(mAP_BEV._extract_relevant_tensor_info)

    @staticmethod
    def _extract_relevant_tensor_info(
        module: nn.Module,
        inputs: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred, gt = inputs
        gt_info = gt["gt_info"].squeeze(0)
        pred_labels = pred[:, 0]
        pred_class_probits = pred[:, 1]
        pred_boxes_3d = pred[:, PRED_BOUNDING_BOX_3D_SLICE]
        pred_boxes_bev = project_3d_boxes_to_bev(pred_boxes_3d)
        gt_boxes_3d = gt_info[:, GT_BOUNDING_BOX_3D_SLICE]
        gt_boxes_bev = project_3d_boxes_to_bev(gt_boxes_3d)
        return (pred_class_probits, pred_boxes_bev, pred_labels), gt_boxes_bev

    def forward(
        self, pred: tuple[torch.Tensor, torch.Tensor, torch.Tensor], gt: torch.Tensor
    ) -> torch.Tensor:
        return mAP_pascal_voc(pred=pred, gt=gt)
