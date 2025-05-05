import torch
from torch import nn


class OutputHeads(nn.Module):
    rpn_out_channels = 256
    output_features = 256

    def __init__(self, pool_size: list[int, int], num_classes: int) -> None:
        """
        Regression and classification heads for the Faster RCNN module.

        Args:
            pool_size: Height and width of the feature maps after applying RoI align.
            num_classes: Number of classes we are trying to predict + 1 for background.
        """
        super().__init__()
        self.fc1 = nn.Linear(
            in_features=self.rpn_out_channels * pool_size[0] * pool_size[1],
            out_features=self.output_features,
        )
        self.classification_head = nn.Linear(
            in_features=self.output_features, out_features=num_classes
        )
        self.regression_head = nn.Linear(
            in_features=self.output_features, out_features=4 * num_classes
        )

    def forward(
        self, pooled_proposals_per_feature_map: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        num_proposals = pooled_proposals_per_feature_map.shape[0]
        intermediary = self.fc1(
            pooled_proposals_per_feature_map.view(num_proposals, -1)
        )
        class_logits = self.classification_head(intermediary)
        bounding_box_deltas = self.regression_head(intermediary)

        return class_logits, bounding_box_deltas
