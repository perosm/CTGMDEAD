import torch
from torch import nn


class FasterRCNNHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        pool_size: int = 7,
        fc_hidden_dim: int = 1024,
        num_classes: int = 3,
    ):
        super().__init__()
        """
        Implementation of the Faster R-CNN classification and regression head.
        
        Args:

        """
        self.in_channels = in_channels
        self.pool_size = pool_size
        self.fc_hidden_dim = fc_hidden_dim
        self.num_classes = num_classes

        self.fc1 = nn.Linear(
            self.in_channels * self.pool_size * self.pool_size, self.fc_hidden_dim
        )
        self.fc2 = nn.Linear(self.fc_hidden_dim, self.fc_hidden_dim)
        self.classification_head = nn.Linear(self.fc_hidden_dim, self.num_classes)
        self.regression_head = nn.Linear(self.fc_hidden_dim, self.num_classes * 4)

    def forward(self, pooled_proposals: dict[str, torch.Tensor]) -> torch.Tensor:
        class_logits, bounding_box_deltas = [], []
        for pooled_proposal in pooled_proposals.values():
            intermediary = self.fc2(self.fc1(pooled_proposal))
            class_logit = self.classification_head(intermediary)
            bounding_box_delta = self.regression_head(intermediary)
            class_logits.append(class_logit)
            bounding_box_deltas.append(bounding_box_delta)

        return class_logits, bounding_box_deltas
