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


class UnetInputReconstructionDecoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.in_channels = kwargs.get("in_channels", 512)
        self.channel_scale_factors = kwargs.get("channel_scale_factors", [2, 4, 8, 8])
        self.out_channels = kwargs.get("out_channels", 3)
        self.main_decoder = kwargs.get("main_decoder", False)
        self.layer1 = UnetLayer(
            self.in_channels, self.in_channels // self.channel_scale_factors[0]
        )
        self.layer2 = UnetLayer(
            self.in_channels // self.channel_scale_factors[0],
            self.in_channels // self.channel_scale_factors[1],
        )
        self.layer3 = UnetLayer(
            self.in_channels // self.channel_scale_factors[1],
            self.in_channels // self.channel_scale_factors[2],
        )
        self.layer4 = UnetLayer(
            self.in_channels // self.channel_scale_factors[2],
            self.in_channels // self.channel_scale_factors[3],
        )
        self.conv = nn.Conv2d(
            self.in_channels // self.channel_scale_factors[3],
            self.out_channels,
            kernel_size=1,
        )

    def forward(
        self, encoder_outputs: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        fpn_outputs = {}
        fpn_outputs["fpn3"] = self.layer1(encoder_outputs["e4"], encoder_outputs["e3"])
        fpn_outputs["fpn2"] = self.layer2(fpn_outputs["fpn3"], encoder_outputs["e2"])
        fpn_outputs["fpn1"] = self.layer3(fpn_outputs["fpn2"], encoder_outputs["e1"])
        fpn_outputs["fpn0"] = self.layer4(fpn_outputs["fpn1"], encoder_outputs["e0"])
        input = self.conv(fpn_outputs["fpn0"])

        if self.main_decoder:
            return fpn_outputs, F.interpolate(input, scale_factor=2, mode="bilinear")

        return F.interpolate(input, scale_factor=2, mode="bilinear")
