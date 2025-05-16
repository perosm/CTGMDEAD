import torch
from torch import nn
from model.resnet import ResNet
from utils.shared.enums import TaskEnum


class MultiTaskNetwork(nn.Module):

    def __init__(
        self,
        encoder: ResNet,
        depth_decoder: nn.Module,
        road_detection_decoder: nn.Module | None = None,
        heads_and_necks: dict[str, nn.Module] | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.depth_decoder = depth_decoder
        self.road_detection_decoder = road_detection_decoder
        self.heads_and_necks = heads_and_necks

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        task_outputs = {}
        encoder_outputs = self.encoder(x)
        fpn_outputs, task_outputs[TaskEnum.depth] = self.depth_decoder(encoder_outputs)

        if self.road_detection_decoder:
            task_outputs[TaskEnum.road_detection] = self.road_detection_decoder(
                encoder_outputs
            )

        if self.heads_and_necks:
            for task in self.heads_and_necks:
                task_outputs[task] = self.heads_and_necks[task](fpn_outputs)

        return task_outputs
