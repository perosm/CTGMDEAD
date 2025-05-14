import torch

from typing import Any


class PredictionPostprocessor:
    def __init__(self, per_task_postprocessing_funcs: dict[str, Any]) -> None:
        super().__init__()
        self.per_task_postprocessing_funcs = per_task_postprocessing_funcs

    def __call__(
        self, per_task_prediction: dict[str, torch.Tensor | dict[str, tuple]]
    ) -> dict[str, torch.Tensor]:
        """
        Postprocesses outputs of the network.
        """
        processed_per_task_prediction = per_task_prediction
        for task, postprocessing_function in self.per_task_postprocessing_funcs.items():
            processed_per_task_prediction[task] = postprocessing_function(
                per_task_prediction[task]
            )

        return processed_per_task_prediction
