import torch
from torch import nn
from model.resnet import ResNet
from utils.shared.enums import TaskEnum


class MultiTaskNetwork(nn.Module):
    def __init__(
        self, encoder: ResNet, decoder: nn.Module, heads_and_necks: dict[str, nn.Module]
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.heads_and_necks = heads_and_necks

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        task_outputs = {}
        encoder_outputs = self.encoder(x)
        fpn_outputs, task_outputs[TaskEnum.depth] = self.decoder(encoder_outputs)

        for task in self.heads_and_necks:
            task_outputs[task] = self.heads_and_necks[task](fpn_outputs)

        return task_outputs
