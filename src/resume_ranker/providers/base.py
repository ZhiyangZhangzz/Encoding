from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterable, List, Optional
import pathlib


class BiEncoder(ABC):
    def __init__(self, model_name: str, model_cache_dir: Optional[str] = None):
        self.model_name = model_name
        self.model_cache_dir = pathlib.Path(model_cache_dir) if model_cache_dir else None
        self.model = None

    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def encode(self, texts: Iterable[str], batch_size: int = 32, normalize: bool = True) -> List[list]:
        ...
        

class CrossEncoderBase(ABC):
    def __init__(self, model_name: str, model_cache_dir: Optional[str] = None):
        self.model_name = model_name
        self.model_cache_dir = pathlib.Path(model_cache_dir) if model_cache_dir else None
        self.model = None

    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def predict(self, pairs: Iterable[tuple[str, str]], batch_size: int = 32) -> List[float]:
        ...
