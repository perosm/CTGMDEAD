import torch
import numpy as np
import cv2

from utils.shared.visualizer import VisualizerStrategy
from utils.shared.enums import TaskEnum
from utils.object_detection_3d.utils import (
    project_3d_boxes_to_image,
    project_3d_boxes_to_bev,
)
from utils.shared.enums import ObjectDetectionEnum
from torchvision.utils import draw_bounding_boxes

LABEL_TO_NAME_MAPPING = {
    1: "Car",
    2: "Person",
    3: "Tram",
}


class Visualizer(VisualizerStrategy):
    task = TaskEnum.object_detection
    pred_box_info_2d_slice = slice(0, 6)
    pred_bounding_box_3d_slice = slice(6, 13)
    gt_bounding_box_2d_slice = slice(
        ObjectDetectionEnum.box_2d_left, ObjectDetectionEnum.box_2d_bottom + 1
    )
    gt_bounding_box_3d_slice = slice(
        ObjectDetectionEnum.height, ObjectDetectionEnum.rotation_y + 1
    )

    def __init__(self):
        pass

    def visualize(
        self,
        pred: torch.Tensor,
        gt: dict[str, torch.Tensor],
        image: torch.Tensor,
    ) -> dict[str, np.ndarray]:
        gt_info = gt["gt_info"].squeeze(0).cpu()
        image_od_2d = self._visualize_2d(
            pred=pred[:, self.pred_box_info_2d_slice].to(torch.int64),
            gt=gt_info[:, self.gt_bounding_box_2d_slice].to(torch.int64),
            image=image.clone().to(torch.uint8),
        )

        projection_matrix = gt["projection_matrix"].squeeze(0).cpu()
        image_od_3d = self._visualize_3d(
            pred=pred[:, self.pred_bounding_box_3d_slice],
            gt=gt_info[:, self.gt_bounding_box_3d_slice],
            image=image.clone().permute(1, 2, 0).numpy().astype(np.uint8),
            projection_matrix=projection_matrix,
        )

        image_od_3d_bev = self._visualize_3d_bev(
            pred=pred[:, self.pred_bounding_box_3d_slice],
            gt=gt_info[:, self.gt_bounding_box_3d_slice],
            image_shape=image.shape,
        )

        return {
            f"{self.task}_2d": image_od_2d,
            f"{self.task}_3d": image_od_3d,
            f"{self.task}_bev": image_od_3d_bev,
        }

    def _visualize_2d(
        self, pred: torch.Tensor, gt: torch.Tensor, image: torch.Tensor
    ) -> np.ndarray:
        labels = pred[:, 0]
        boxes_2d = pred[:, 2:]
        keep = (boxes_2d[:, 0] < boxes_2d[:, 2]) & (boxes_2d[:, 1] < boxes_2d[:, 3])
        boxes_2d = boxes_2d[keep]
        labels = labels[keep]

        image = draw_bounding_boxes(
            image=image,
            boxes=boxes_2d,
            colors=(255, 0, 0),
            labels=[LABEL_TO_NAME_MAPPING[label.item()] for label in labels],
        )

        image = draw_bounding_boxes(image=image, boxes=gt, colors=(0, 255, 0))

        return image.permute(1, 2, 0).numpy()

    def _visualize_3d(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        image: torch.Tensor,
        projection_matrix: torch.Tensor,
    ) -> dict[str, np.ndarray]:
        # Solves error: Layout of the output array img is incompatible with cv::Mat
        # https://stackoverflow.com/questions/26998223/what-is-the-difference-between-contiguous-and-non-contiguous-arrays
        image = np.ascontiguousarray(image)
        projected_pred_boxes = (
            project_3d_boxes_to_image(
                boxes_3d_info=pred, projection_matrix=projection_matrix
            )
            .numpy()
            .astype(np.int32)
        )
        projected_boxes = (
            project_3d_boxes_to_image(
                boxes_3d_info=gt, projection_matrix=projection_matrix
            )
            .numpy()
            .astype(np.int32)
        )
        projected_heights = (
            self._fetch_projected_height(
                box_3d_info=gt, projection_matrix=projection_matrix
            )
            .numpy()
            .astype(np.int32)
        )

        # print("Pred: ", pred)
        # print("Gt:", gt)
        for projected_pred_box in projected_pred_boxes:
            Visualizer.draw_3d_box(
                image=image, box_3d=projected_pred_box, color=(255, 0, 0)
            )

        for projected_box, projected_height in zip(projected_boxes, projected_heights):
            Visualizer.draw_3d_box(image=image, box_3d=projected_box, color=(0, 255, 0))
            start_point = tuple(projected_height[:2].tolist())
            end_point = tuple(projected_height[2:].tolist())
            cv2.line(image, start_point, end_point, color=(0, 255, 0), thickness=1)

        return image

    def _visualize_3d_bev(
        self, pred: torch.Tensor, gt: torch.Tensor, image_shape: tuple[int, int]
    ) -> np.ndarray:
        C, H, W = image_shape
        image = np.zeros((H, W, C))  # (80, 90))
        pred_boxes_bev = project_3d_boxes_to_bev(boxes_3d_info=pred).numpy()
        gt_boxes_bev = project_3d_boxes_to_bev(boxes_3d_info=gt).numpy()

        def to_pixel(x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """
            ________z (u)
            |
            |
            |
            x (v)

            z coordinate corresponds to
            """

            # x: [-40, 40] -> [0, 80]
            v = (((x + 40) / 80.0) * (W - 1)).astype(np.int32)
            # z: [0, 90]
            u = ((z / 90.0) * W).astype(np.int32)
            v = np.clip(v, 0, W)
            u = np.clip(u, 0, W)

            return u, v

        pred_u, pred_v = to_pixel(pred_boxes_bev[:, 0], pred_boxes_bev[:, 1])
        gt_u, gt_v = to_pixel(gt_boxes_bev[:, 0], gt_boxes_bev[:, 1])

        for box_corners in np.stack((pred_u, pred_v), axis=-1):
            box_corners = box_corners.reshape(-1, 1, 2)
            cv2.polylines(
                image, [box_corners], isClosed=True, color=(255, 0, 0), thickness=1
            )

        for box_corners in np.stack((gt_u, gt_v), axis=-1):
            box_corners = box_corners.reshape(-1, 1, 2)
            cv2.polylines(
                image, [box_corners], isClosed=True, color=(0, 255, 0), thickness=1
            )
        return image

    @staticmethod
    def draw_3d_box(
        image: np.ndarray, box_3d: np.ndarray, color: tuple[int, int, int]
    ) -> np.ndarray:
        edges = [
            # bottom
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            # top
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            # vertical
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]

        for start, end in edges:
            start_point = tuple(box_3d[:2, start].astype(int).tolist())
            end_point = tuple(box_3d[:2, end].astype(int).tolist())
            cv2.line(image, start_point, end_point, color=color, thickness=1)

    def _fetch_projected_height(
        self, box_3d_info: torch.Tensor, projection_matrix: torch.Tensor
    ):
        gt_H = box_3d_info[:, 0].float()
        gt_distance = box_3d_info[:, 5].float()
        focal_x = projection_matrix[0, 0].float()
        h = (focal_x * gt_H / gt_distance).round().int()

        object_center = box_3d_info[:, 3:6].float()
        ones = torch.ones(object_center.shape[0], 1, device=object_center.device)
        object_center_homogeneous = torch.cat([object_center, ones], dim=1)

        projected = (projection_matrix[:3].float() @ object_center_homogeneous.T).T
        projected_center = (projected[:, :2] / projected[:, 2:3]).round().int()

        bottom_points = projected_center.clone()
        bottom_points[:, 1] -= h

        return torch.cat([bottom_points, projected_center], dim=1)
