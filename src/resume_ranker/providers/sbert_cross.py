from __future__ import annotations
from typing import Iterable, List, Optional
from .base import CrossEncoderBase
from .exceptions import ModelNotLoadedError


class SbertCrossEncoder(CrossEncoderBase):
    """
    Sentence-Transformers CrossEncoder 封装
    """
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        model_cache_dir: Optional[str] = None,
        device: Optional[str] = None,
    ):
        super().__init__(model_name, model_cache_dir)
        self.device = device

    def load(self) -> None:
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(
            self.model_name,
            cache_folder=str(self.model_cache_dir) if self.model_cache_dir else None,
            device=self.device,
        )

    def predict(self, pairs: Iterable[tuple[str, str]], batch_size: int = 32) -> List[float]:
        if self.model is None:
            raise ModelNotLoadedError("Cross-encoder 未加载，请先调用 load()。")
        return self.model.predict(list(pairs), batch_size=batch_size).tolist()
