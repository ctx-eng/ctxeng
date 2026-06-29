from __future__ import annotations

from typing import List, Optional


class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None

    @property
    def available(self) -> bool:
        try:
            import sentence_transformers  # noqa: F401
            return True
        except ImportError:
            return False

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.model_name)

    def encode(self, texts: List[str]) -> List[List[float]]:
        self._lazy_load()
        embeddings = self._model.encode(texts)
        return embeddings.tolist()

    def encode_query(self, query: str) -> List[float]:
        return self.encode([query])[0]

    @property
    def dimension(self) -> Optional[int]:
        if not self.available:
            return None
        self._lazy_load()
        return self._model.get_sentence_embedding_dimension()
