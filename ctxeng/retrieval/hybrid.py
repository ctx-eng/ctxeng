from __future__ import annotations

from typing import List, Optional

from ctxeng.models import MemoryItem
from ctxeng.retrieval.embeddings import EmbeddingModel
from ctxeng.stores.base import ContextStore

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    if HAS_NUMPY:
        na = np.array(a)
        nb = np.array(b)
        denom = np.linalg.norm(na) * np.linalg.norm(nb)
        if denom == 0.0:
            return 0.0
        return float(np.dot(na, nb) / denom)
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


class BM25Retriever:
    def __init__(self) -> None:
        self._corpus: List[str] = []
        self._bm25 = None

    def index(self, texts: List[str]) -> None:
        self._corpus = texts
        if not texts:
            self._bm25 = None
            return
        try:
            from rank_bm25 import BM25Okapi
            tokenized = [_tokenize(t) for t in texts]
            self._bm25 = BM25Okapi(tokenized)
        except ImportError:
            self._bm25 = None

    def search(self, query: str) -> List[float]:
        if self._bm25 is None:
            return [0.0] * len(self._corpus)
        tokenized_query = _tokenize(query)
        return self._bm25.get_scores(tokenized_query).tolist()


class HybridRetriever:
    def __init__(
        self,
        store: ContextStore,
        alpha: float = 0.5,
        embedder: Optional[EmbeddingModel] = None,
    ) -> None:
        self.store = store
        self.alpha = alpha
        self.embedder = embedder or EmbeddingModel()
        self._bm25 = BM25Retriever()

    def search(self, user_id: str, query: str, top_k: int = 10) -> List[MemoryItem]:
        candidates = self.store.list(user_id)

        if not query:
            for c in candidates:
                c.score = 0.5
            return candidates[:top_k]

        texts = [c.text for c in candidates]
        n = len(candidates)

        dense_scores: List[float]
        if self.embedder.available and n > 0:
            query_emb = self.embedder.encode_query(query)
            candidate_embs = self.embedder.encode(texts)
            dense_scores = [
                _cosine_similarity(query_emb, cand_emb) for cand_emb in candidate_embs
            ]
        else:
            dense_scores = [0.0] * n

        self._bm25.index(texts)
        sparse_scores = self._bm25.search(query)

        query_lower = query.lower()
        keyword_scores = [1.0 if query_lower in t.lower() else 0.0 for t in texts]

        for i in range(n):
            combined = self.alpha * dense_scores[i] + (1 - self.alpha) * sparse_scores[i]
            combined += keyword_scores[i]
            candidates[i].score = combined

        ranked = sorted(candidates, key=lambda x: x.score, reverse=True)
        return ranked[:top_k]
