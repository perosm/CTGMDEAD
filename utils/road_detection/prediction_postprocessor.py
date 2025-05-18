import torch


class PredictionPostprocessor:
    def __init__(self):
        super().__init__()

    def __call__(
        self, pred: dict[str, tuple[torch.Tensor, ...]]
    ) -> dict[str, tuple[torch.Tensor, ...]]:
        return pred  # TODO: Should there be any postprocessing ?
