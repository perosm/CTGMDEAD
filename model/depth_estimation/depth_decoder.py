import torch
from torch import nn
import torch.nn.functional as F


class UnetLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upconv1 = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
            bias=False,
        )
        self.conv1 = nn.Conv2d(
            2 * out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        x = self.upconv1(x)
        x = torch.cat([e, x], dim=1)
        return self.conv2(self.relu(self.conv1(x)))


class UnetDepthDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 512,
        channel_scale_factors: list[int] = [2, 4, 8, 8],
        out_channels: int = 1,
    ):
        super().__init__()
        self.layer1 = UnetLayer(in_channels, in_channels // channel_scale_factors[0])
        self.layer2 = UnetLayer(
            in_channels // channel_scale_factors[0],
            in_channels // channel_scale_factors[1],
        )
        self.layer3 = UnetLayer(
            in_channels // channel_scale_factors[1],
            in_channels // channel_scale_factors[2],
        )
        self.layer4 = UnetLayer(
            in_channels // channel_scale_factors[2],
            in_channels // channel_scale_factors[3],
        )
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels // channel_scale_factors[3], out_channels, kernel_size=1
            ),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, encoder_outputs: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        fpn_outputs = {}
        fpn_outputs["fpn3"] = self.layer1(encoder_outputs["e4"], encoder_outputs["e3"])
        fpn_outputs["fpn2"] = self.layer2(fpn_outputs["fpn3"], encoder_outputs["e2"])
        fpn_outputs["fpn1"] = self.layer3(fpn_outputs["fpn2"], encoder_outputs["e1"])
        fpn_outputs["fpn0"] = self.layer4(fpn_outputs["fpn1"], encoder_outputs["e0"])
        depth = self.conv(fpn_outputs["fpn0"])

        return fpn_outputs, F.interpolate(depth, scale_factor=2, mode="bilinear")
