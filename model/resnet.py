import torch
from torch import nn

from typing import Union, List
import torchvision


# TODO: will BottleNeck block be used since we are going with ResNets < ResNet50 ?
class BottleneckBlock(nn.Module):
    channel_expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        stride: int,
        downsample_flag: bool = False,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(in_channels)
        out_channels = in_channels * self.channel_expansion
        self.conv3 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if downsample_flag:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> None:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        if self.downsample:
            return self.relu(self.bn3(self.conv3(out)) + self.downsample(x))
        else:
            return self.relu(self.bn3(self.conv3(out)) + x)


class BasicBlock(nn.Module):
    channel_expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        downsample_flag: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        if downsample_flag:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=2,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        if self.downsample:
            return self.relu(self.bn2(self.conv2(out)) + self.downsample(x))
        else:
            return self.relu(self.bn2(self.conv2(out)) + x)


class ResNet(nn.Module):
    def __init__(
        self,
        blocks_per_layer: List[int],
        block: Union[BottleneckBlock, BasicBlock],
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._configure_layer(
            num_blocks=blocks_per_layer[0],
            in_channels=64,
            out_channels=64,
            stride=1,
            block=block,
            downsample=False,
        )
        self.layer2 = self._configure_layer(
            num_blocks=blocks_per_layer[1],
            in_channels=64,
            out_channels=128,
            stride=2,
            block=block,
            downsample=True,
        )
        self.layer3 = self._configure_layer(
            num_blocks=blocks_per_layer[2],
            in_channels=128,
            out_channels=256,
            stride=2,
            block=block,
            downsample=True,
        )
        self.layer4 = self._configure_layer(
            num_blocks=blocks_per_layer[3],
            in_channels=256,
            out_channels=512,
            stride=2,
            block=block,
            downsample=True,
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=512, out_features=1000)

    def _configure_layer(
        self,
        num_blocks: int,
        stride: int,
        in_channels: int,
        out_channels: int,
        downsample: bool,
        block: Union[BottleneckBlock, BasicBlock],
    ) -> nn.Sequential:
        blocks = []
        if downsample:
            blocks.append(
                block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    downsample_flag=True,
                )
            )
        for _ in range(num_blocks - len(blocks)):
            blocks.append(
                block(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    stride=1,
                    downsample_flag=False,
                )
            )

        return nn.Sequential(*blocks)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoder_output = {}
        encoder_output["e0"] = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(encoder_output["e0"])
        encoder_output["e1"] = self.layer1(out)
        encoder_output["e2"] = self.layer2(encoder_output["e1"])
        encoder_output["e3"] = self.layer3(encoder_output["e2"])
        encoder_output["e4"] = self.layer4(encoder_output["e3"])

        return encoder_output


def ResNet18(pretrained: bool = True) -> ResNet:
    """
    Args:
        - pretrained: flag to indicate whether to load imagenet pretrained ResNet18.

    Returns:
        - model:
    """
    model = ResNet(blocks_per_layer=[2, 2, 2, 2], block=BasicBlock)
    if pretrained:
        resnet_18 = torchvision.models.resnet18(
            weights="ResNet18_Weights.IMAGENET1K_V1"
        )
        model.load_state_dict(resnet_18.state_dict(), strict=True)

    return model


def ResNet50(pretrained: bool = True):
    model = ResNet(blocks_per_layer=[2, 2, 2, 2], block=BasicBlock)
    if pretrained:
        resnet_18 = torchvision.models.resnet18(
            weights="ResNet18_Weights.IMAGENET1K_V1"
        )
        model.load_state_dict(resnet_18.state_dict(), strict=True)


if __name__ == "__main__":
    device = "cuda"
    # resnet_18 = torchvision.models.resnet18(
    #     weights="ResNet18_Weights.IMAGENET1K_V1"
    # ).to(device)
    # model = ResNet18(pretrained=True).to(device)

    # x = torch.zeros(1, 3, 256, 256).to(device)
    # with torch.no_grad():
    #     gt = resnet_18(x)
    #     pred = model(x)

    # assert (gt - pred).sum() == 0, "They are not the same!"
