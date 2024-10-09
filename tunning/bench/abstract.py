from pathlib import Path
from abc import ABC, abstractmethod
from typing import Tuple


class AbstractBench(ABC):
    @abstractmethod
    def run(self, conf_path: Path):
        pass

    @abstractmethod
    def cleanup(self):
        pass

    @abstractmethod
    def parse_result(self, task_id: int, conf_path: Path) -> Tuple[float, float]:
        pass
