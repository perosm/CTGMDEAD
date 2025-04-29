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

    def forward(self, x):
        e = self.encoder(x)
        out = {}

        return out
