import torch
from torch import nn
from torchvision.ops import roi_align


class ROINetwork(nn.Module):
    def __init__(
        self, image_size: tuple[int, int], pool_output_size: tuple[int, int] = (7, 7)
    ):
        super().__init__()
        self.image_size = image_size
        self.pool_output_size = pool_output_size

    def forward(
        self,
        fpn_feature_map_outputs: dict[str, torch.Tensor],
        proposals_per_feature_map: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        pooled_proposal_feature_map = {}
        fpn_names = proposals_per_feature_map.keys()
        for fpn_name in fpn_names:
            H = fpn_feature_map_outputs[fpn_name].shape[-2]
            spatial_scale = H / self.image_size[0]
            pooled_feature_map = roi_align(
                input=fpn_feature_map_outputs[fpn_name],
                boxes=proposals_per_feature_map[fpn_name],
                output_size=self.pool_output_size,
                spatial_scale=spatial_scale,
                sampling_ratio=4,
                aligned=False,
            )
            pooled_proposal_feature_map[fpn_name] = pooled_feature_map

        return pooled_proposal_feature_map
