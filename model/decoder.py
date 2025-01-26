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

    def forward(self, x, e) -> torch.Tensor:
        x = self.upconv1(x)
        x = torch.cat([e, x], dim=1)
        return self.conv2(self.relu(self.conv1(x)))


class UnetDecoder(nn.Module):
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
        self.conv = nn.Conv2d(
            in_channels // channel_scale_factors[3], out_channels, kernel_size=1
        )

    def forward(self, e0, e1, e2, e3, e4) -> torch.Tensor:
        x = self.layer1(e4, e3)
        x = self.layer2(x, e2)
        x = self.layer3(x, e1)
        x = self.layer4(x, e0)
        x = self.conv(x)

        return F.interpolate(x, scale_factor=2, mode="bilinear")


if __name__ == "__main__":
    e0 = torch.zeros((1, 64, 128, 128))
    e1 = torch.zeros((1, 64, 64, 64))
    e2 = torch.zeros((1, 128, 32, 32))
    e3 = torch.zeros((1, 256, 16, 16))
    e4 = torch.zeros((1, 512, 8, 8))

    decoder = UnetDecoder()
    y = decoder(e4, e3, e2, e1, e0)
    gt = torch.zeros((1, 1, 256, 256))

    assert (
        y.shape == gt.shape
    ), "Output image should be of same dimensionality as input image!"
