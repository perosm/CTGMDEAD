import os
import abc
import pathlib
from utils.shared.aggregators.Aggregator import Aggregator


class Saver(abc.ABC):
    def __init__(self, aggregator: Aggregator, save_dir: pathlib.Path):
        self.aggregator = aggregator
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    @abc.abstractmethod
    def save() -> None:
        pass
