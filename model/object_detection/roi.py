import torch
from torch import nn
from torchvision.ops import MultiScaleRoIAlign


class ROINetwork(nn.Module):
    def __init__(
        self,
        image_size: tuple[int, int],
        pool_output_size: tuple[int, int] = (7, 7),
        feature_map_names=["fpn0", "fpn1", "fpn2", "fpn3"],
        sampling_ratio: int = 4,
    ):
        super().__init__()
        self.image_size = image_size
        self.pool_output_size = pool_output_size
        self.roi_pool = MultiScaleRoIAlign(
            featmap_names=feature_map_names,
            output_size=7,
            sampling_ratio=2,
        )

    def forward(
        self,
        fpn_feature_map_outputs: dict[str, torch.Tensor],
        proposals: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        fpn_feature_map_outputs = dict(sorted(fpn_feature_map_outputs.items()))
        pooled_proposals = self.roi_pool(
            x=fpn_feature_map_outputs,
            boxes=[proposals],
            image_shapes=[self.image_size],
        )

        return pooled_proposals
