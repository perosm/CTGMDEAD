import torch
from torch import nn
from model.object_detection.anchor_generator import AnchorGenerator


class DetectionHead(nn.Module):
    in_channels = 256
    out_channels = 1

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.out_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class RegressionHead(nn.Module):
    in_channels = 256
    out_channels = 4

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.out_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class RPNHead(nn.Module):
    out_channels = 256

    def __init__(self, num_channels_per_feature_map: list[int]):
        super().__init__()
        self.detection_head = DetectionHead()
        self.regression_head = RegressionHead()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        cls = self.detection_head(x)
        reg = self.regression_head(x)

        return cls, reg


class RegionProposalNetwork(nn.Module):

    def __init__(
        self,
        num_channels_per_feature_map: list[int],
        anchor_generator: AnchorGenerator,
    ) -> None:
        """
        Region Proposal Network (RPN) is used to predict whether an object exists
        """
        super().__init__()
        self.conv = nn.ModuleDict(
            {
                f"{i}": nn.Conv2d(
                    in_channels=num_channels, out_channels=self.out_channels
                )
                for i, num_channels in enumerate(num_channels_per_feature_map)
            }
        )
        self.anchor_generator = anchor_generator
        self.rpn_head = RPNHead()

    def forward(
        self,
        input_image: torch.Tensor,
        encoder_outputs: dict[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        classifications_per_feature_map, bbox_regressions_per_feature_map = [], []
        for i, encoder_output in enumerate(encoder_outputs):
            intermediary = self.conv[i](encoder_output)
            bbox_cls, bbox_reg = self.rpn_head(intermediary)
            classifications_per_feature_map.append(bbox_cls)
            bbox_regressions_per_feature_map.append(bbox_reg)

        return classifications_per_feature_map, bbox_regressions_per_feature_map


if __name__ == "__main__":
    num_channels_per_feature_map = [64, 128, 256, 512]
    anchor_generator = AnchorGenerator(
        sizes=(  # TODO: which anchor sizes to choose?
            (512),
            (256),
            (128),
            (64, 32),
        ),
        aspect_ratios=(
            (0.5, 1.0, 2.0),
            (0.5, 1.0, 2.0),
            (0.5, 1.0, 2.0),
            (0.5, 1.0, 2.0),
        ),
    )
    input_image = torch.zeros((2, 3, 256, 1184))
    feature_maps = (
        torch.zeros((512, 16, 74)),
        torch.zeros((256, 32, 148)),
        torch.zeros((128, 64, 296)),
        torch.zeros((64, 128, 592)),
    )
    rpn = RegionProposalNetwork(
        num_channels_per_feature_map=num_channels_per_feature_map,
        anchor_generator=anchor_generator,
    )
    rpn(input_image, feature_maps)
