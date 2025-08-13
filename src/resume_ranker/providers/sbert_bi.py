from __future__ import annotations
from typing import Iterable, List, Optional
from .base import BiEncoder
from .exceptions import ModelNotLoadedError


class SBERTBiEncoder(BiEncoder):
    """
    SentenceTransformer bi-encoder 封装
    """
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        model_cache_dir: Optional[str] = None,
        device: Optional[str] = None,
    ):
        super().__init__(model_name, model_cache_dir)
        self.device = device  # e.g. "cuda" / "cpu"

    def load(self) -> None:
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(
            self.model_name,
            cache_folder=str(self.model_cache_dir) if self.model_cache_dir else None,
            device=self.device,
        )

    def encode(self, texts: Iterable[str], batch_size: int = 32, normalize: bool = True) -> List[list]:
        if self.model is None:
            raise ModelNotLoadedError("Bi-encoder 未加载，请先调用 load()。")
        return self.model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        ).tolist()
