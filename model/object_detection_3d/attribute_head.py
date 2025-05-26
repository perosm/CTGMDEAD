import torch
from torch import nn


class AttributeHead(nn.Module):
    num_keypoints = 9
    size_head_out_features = 3
    yaw_head_out_features = 2
    keypoints_head_out_features = 2 * num_keypoints  # 2 for x and y each

    def __init__(
        self,
        num_conv_layers: int,
        num_channels: int,
        num_fc_layers: int,
        rpn_output_channels: int,
        pool_output_size: tuple[int, int],
        fc_features: int,
    ):
        """
        Three dimensional attribute head is used to predict:
            - 1. Physical size m = (width, height, length)
            - 2. Yaw angle a = (sin(theta), cos(theta))
            - 3. 2D keypoints (i.e. projected center and corners of 3D bounding box)

               (5)-----(4)
               /|       /|
              / |      / |
            (6)-----(7)  |
             |  | (8)|   |
             | (1)---|--(0)
             | /     |  /
             |/      | /
            (2)-----(3)

            Numbers (0)-(7) represent vertices, and number (8) represents the
            center of the 3D bounding box.

            Args:
                num_conv_laysrs: Number of shared convolution layers.
                num_channels: Number of channels in the intermediary conv layers.
                num_fc_layers: Number of shared fully connected layers.
                rpn_output_channels: Output channel size from the RPN head.
                pool_output_size: Tuple representing height an of the pooled proposals.
                fc_features: Size of the intermediary fully connected layers.
        """
        super().__init__()
        self.num_conv_layers = num_conv_layers
        self.num_channels = num_channels
        self.num_fc_layers = num_fc_layers
        self.rpn_output_channels = rpn_output_channels
        self.pool_output_size = pool_output_size
        self.fc_features = fc_features
        self.convs = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=(
                        self.rpn_output_channels if i == 0 else self.num_channels
                    ),
                    out_channels=self.num_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.BatchNorm2d(num_features=self.num_channels),
                nn.ReLU(inplace=True),
            )
            for i in range(self.num_conv_layers)
        )

        self.fcs = nn.ModuleList(
            nn.Sequential(
                nn.Linear(
                    in_features=(
                        self.num_channels
                        * self.pool_output_size[0]
                        * self.pool_output_size[1]
                        if i == 0
                        else self.fc_features
                    ),
                    out_features=self.fc_features,
                ),
                nn.ReLU(inplace=True),
            )
            for i in range(self.num_fc_layers)
        )

        self.size_head = nn.Sequential(
            nn.Linear(
                in_features=self.fc_features, out_features=self.size_head_out_features
            ),
            nn.ReLU(inplace=True),
        )
        self.yaw_head = nn.Linear(
            in_features=self.fc_features, out_features=self.yaw_head_out_features
        )
        self.keypoints_head = nn.Sequential(
            nn.Linear(
                in_features=self.fc_features,
                out_features=self.keypoints_head_out_features,
            ),
            nn.ReLU(inplace=True),
        )

    def _forward_convs(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = conv(x)

        return x

    def _forward_fcs(self, x: torch.Tensor) -> torch.Tensor:
        for fc in self.fcs:
            x = fc(x)

        return x

    def forward(self, pooled_proposals: torch.Tensor) -> torch.Tensor:
        intermediary = self._forward_fcs(
            torch.flatten(self._forward_convs(pooled_proposals), start_dim=1)
        )
        size = self.size_head(intermediary)
        yaw = self.yaw_head(intermediary)
        keypoints = self.keypoints_head(intermediary).view(-1, 2, self.num_keypoints)

        return size, yaw, keypoints
