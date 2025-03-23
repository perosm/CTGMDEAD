import torch
from torch import nn
from model.resnet import ResNet


class MultiTaskNetwork(nn.Module):
    def __init__(self, encoder: ResNet, decoders: dict[str, nn.Module]):
        super().__init__()
        self.encoder = encoder
        self.decoders = nn.ModuleDict(decoders)

    def forward(self, x):
        e = self.encoder(x)
        out = {}
        for task in self.decoders.keys():
            out[task] = self.decoders[task](*e)

        return out
