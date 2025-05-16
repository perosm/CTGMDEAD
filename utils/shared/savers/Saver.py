import abc
import pathlib
from utils.shared.aggregators.Aggregator import Aggregator


class Saver(abc.ABC):
    def __init__(self, aggregator: Aggregator, save_dir: pathlib.Path):
        self.aggregator = aggregator
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

    @abc.abstractmethod
    def save(self) -> None:
        pass
