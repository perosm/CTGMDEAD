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

    def train(self, mode=True):
        """
        # https://docs.pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module.train
        The reason for overriding this function is because heads and necks is a dict
        where keys are strings and values are nn.Modules so we can't fetch if through
        self.children(). This is important because FPN Faster R-CNN module, which is
        stored inside the heads_and_necks dict, operates in a single way when we are training
        a model (i.e. model.train()), and in another way when we are evaluating (i.e. model.eval()).
        In the above provided link you can see the original implementation.
        """
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)

        # Added
        if not self.heads_and_necks:
            return

        for task, module in self.heads_and_necks.items():
            module.train(mode)

        return self

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
