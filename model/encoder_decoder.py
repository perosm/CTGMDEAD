import torch
from torch import nn
from model.resnet import ResNet, ResNet18
from model.decoder import UnetDecoder


class DepthEncoderDecoder(nn.Module):
    def __init__(self, encoder: ResNet, decoder: UnetDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x) -> torch.Tensor:
        e0, e1, e2, e3, e4 = self.encoder(x)
        out = self.decoder(e0, e1, e2, e3, e4)

        return out


class EncoderDecoder(nn.Module):
    def __init__(self, encoder: ResNet, decoder: UnetDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x) -> torch.Tensor:
        e0, e1, e2, e3, e4 = self.encoder(x)
        out = self.decoder(e0, e1, e2, e3, e4)

        return out


if __name__ == "__main__":
    device = "cuda"
    encoder = ResNet18()
    decoder = UnetDecoder()
    autoencoder = DepthEncoderDecoder(encoder=encoder, decoder=decoder).to(device)

    x = torch.zeros((1, 3, 256, 256)).to(device)
    y = autoencoder(x)

    assert x.shape[-2:] == y.shape[-2:], "Height and width are not the same!"
