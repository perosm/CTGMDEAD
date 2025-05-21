import torch
from torch import nn


class AttributeHead(nn.Module):
    def __init__(self):
        """
        Three dimensional attribute head is used to predict:
            - 1. Physical size m = (width, height, length)
            - 2. Yaw angle a = (sin(theta), cos(theta))
            - 3. 2D keypoints (i.e. projected center and corners of 3D bounding box)

        """
        super().__init__()

    def forward(self, x) -> torch.Tensor:
        pass
