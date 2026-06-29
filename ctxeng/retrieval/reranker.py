from __future__ import annotations

from typing import Optional

from ctxeng.models import MemoryItem


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self._model = None

    @property
    def available(self) -> bool:
        try:
            from sentence_transformers import CrossEncoder  # noqa: F401
            return True
        except ImportError:
            return False

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(self.model_name)

    def rerank(
        self, query: str, candidates: list[MemoryItem], top_k: Optional[int] = None
    ) -> list[MemoryItem]:
        self._lazy_load()
        if not candidates:
            return []

        pairs = [(query, c.text) for c in candidates]
        scores = self._model.predict(pairs)

        for i, c in enumerate(candidates):
            c.score = float(scores[i])

        ranked = sorted(candidates, key=lambda x: x.score, reverse=True)
        return ranked[:top_k] if top_k is not None else ranked
