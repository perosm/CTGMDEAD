import torch


class PredictionPostprocessor:
    def __init__(self):
        super().__init__()

    def __call__(
        self, input: dict[str, tuple[torch.Tensor, ...]]
    ) -> dict[str, tuple[torch.Tensor, ...]]:
        return input  # TODO: Should there be any postprocessing ?
