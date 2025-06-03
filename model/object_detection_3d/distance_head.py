import torch
from torch import nn


class DistanceHead(nn.Module):
    # channels:
    # - 0: H - regresses physical height
    # - 1: log_sigma_H - regresses variables of uncertainty
    # - 2: h_rec - regresses reciprocal of the projected physical height H
    # - 3: log_sigma_h_rec - regresses variables of uncertainty
    out_features = 4

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
        Three dimensional distance head recovers distance of objects
        based on the geometry-based distance decomposition.

        Args:
            num_conv_layers: Number of convolutional layers.
            num_channels: Number of channels in the convolutional layers.
            num_fc_layers: Number of fully connected layers.
            rpn_output_channels: Number of channels outputed by the RPN module.
            pool_output_size: Height and width dimensions of the RoI.
            fc_features: Number of features in the fully connected layer.
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
        self.head = nn.Linear(
            in_features=self.fc_features, out_features=self.out_features
        )
        nn.init.normal_(self.head.weight, std=0.001)
        nn.init.constant_(self.head.bias, 0)

    def _forward_convs(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = conv(x)

        return x

    def _forward_fcs(self, x: torch.Tensor) -> torch.Tensor:
        for fc in self.fcs:
            x = fc(x)

        return x

    def forward(self, pooled_proposals: torch.Tensor) -> torch.Tensor:
        return self.head(
            self._forward_fcs(
                torch.flatten(self._forward_convs(pooled_proposals), start_dim=1)
            )
        )
