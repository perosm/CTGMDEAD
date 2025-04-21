import abc

import torch


class Aggregator(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def aggregate_per_batch(
        self, per_batch_values: dict[str, dict[str, torch.Tensor]]
    ) -> None:
        pass

    @abc.abstractmethod
    def _aggregate_per_epoch(self) -> None:
        pass

    @abc.abstractmethod
    def return_aggregated(self) -> dict[str, dict[str, torch.Tensor]]:
        pass
