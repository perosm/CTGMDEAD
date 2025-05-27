import torch
from torch import nn
from typing import Any

from utils.shared.enums import TaskEnum


class PredictionPostprocessor(nn.Module):
    def __init__(self, per_task_postprocessing_funcs: dict[str, Any]) -> None:
        super().__init__()
        self.per_task_postprocessing_funcs = per_task_postprocessing_funcs

    def forward(
        self,
        per_task_prediction: dict[str, torch.Tensor | dict[str, tuple]],
        projection_matrix: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Postprocesses outputs of the network.
        """
        processed_per_task_prediction = {}
        for task, postprocessing_function in self.per_task_postprocessing_funcs.items():
            processed_per_task_prediction[task] = postprocessing_function(
                per_task_prediction[task], projection_matrix
            )

        return processed_per_task_prediction
