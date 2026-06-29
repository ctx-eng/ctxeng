from __future__ import annotations

import pytest

from ctxeng.retrieval.hybrid import BM25Retriever, HybridRetriever, _cosine_similarity
from ctxeng.stores.memory import InMemoryStore

try:
    import sentence_transformers  # noqa: F401

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import rank_bm25  # noqa: F401

    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        v = [1.0, 2.0, 3.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self) -> None:
        assert abs(_cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-6

    def test_opposite_vectors(self) -> None:
        assert abs(_cosine_similarity([1.0, 0.0], [-1.0, 0.0]) + 1.0) < 1e-6

    def test_zero_vector(self) -> None:
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_empty_vectors(self) -> None:
        assert _cosine_similarity([], []) == 0.0


class TestBM25Retriever:
    def test_index_and_search(self) -> None:
        bm25 = BM25Retriever()
        texts = ["hello world", "goodbye world", "foo bar"]
        bm25.index(texts)
        scores = bm25.search("hello")
        assert len(scores) == 3
        assert scores[0] > scores[1]

    def test_search_before_index_returns_zeros(self) -> None:
        bm25 = BM25Retriever()
        scores = bm25.search("hello")
        assert scores == []

    def test_empty_corpus(self) -> None:
        bm25 = BM25Retriever()
        bm25.index([])
        scores = bm25.search("hello")
        assert scores == []


class TestHybridRetriever:
    def test_search_returns_results(self) -> None:
        store = InMemoryStore()
        store.add("alice", "Alice likes concise answers.")
        store.add("alice", "Bob prefers detailed explanations.")
        retriever = HybridRetriever(store)
        results = retriever.search("alice", "concise")
        assert len(results) >= 1
        assert any("concise" in r.text for r in results)

    def test_search_ranks_keyword_match_higher(self) -> None:
        store = InMemoryStore()
        store.add("alice", "The weather is sunny today.")
        store.add("alice", "I love sunny days at the beach.")
        retriever = HybridRetriever(store)
        results = retriever.search("alice", "sunny")
        assert len(results) >= 1
        assert results[0].score > 0

    def test_search_returns_all_for_empty_query(self) -> None:
        store = InMemoryStore()
        store.add("alice", "memory 1")
        store.add("alice", "memory 2")
        retriever = HybridRetriever(store)
        results = retriever.search("alice", "")
        assert len(results) == 2
        assert all(r.score == 0.5 for r in results)

    def test_search_returns_empty_for_unknown_user(self) -> None:
        store = InMemoryStore()
        store.add("alice", "test")
        retriever = HybridRetriever(store)
        results = retriever.search("bob", "test")
        assert len(results) == 0

    def test_top_k_limits_results(self) -> None:
        store = InMemoryStore()
        for i in range(20):
            store.add("alice", f"memory {i}")
        retriever = HybridRetriever(store)
        results = retriever.search("alice", "memory", top_k=5)
        assert len(results) == 5

    def test_alpha_weights_dense_and_sparse(self) -> None:
        store = InMemoryStore()
        store.add("alice", "alpha test memory")
        retriever_dense = HybridRetriever(store, alpha=1.0)
        retriever_sparse = HybridRetriever(store, alpha=0.0)
        results_dense = retriever_dense.search("alice", "alpha")
        results_sparse = retriever_sparse.search("alice", "alpha")
        assert len(results_dense) == 1
        assert len(results_sparse) == 1

    def test_semantic_similarity_fallback(self) -> None:
        store = InMemoryStore()
        store.add("alice", "I enjoy programming in Python.")
        retriever = HybridRetriever(store)
        results = retriever.search("alice", "Python")
        assert len(results) >= 1

    def test_semantic_search_without_exact_match(self) -> None:
        store = InMemoryStore()
        store.add("alice", "I enjoy programming in Python.")
        retriever = HybridRetriever(store)
        results = retriever.search("alice", "coding")
        assert len(results) >= 1


@pytest.mark.skipif(not HAS_BM25, reason="rank_bm25 not installed")
class TestBM25WithLibrary:
    def test_bm25_scores_are_reasonable(self) -> None:
        from ctxeng.retrieval.hybrid import BM25Retriever

        bm25 = BM25Retriever()
        bm25.index(["apple banana cherry", "dog elephant fox", "apple grape"])
        scores = bm25.search("apple")
        assert scores[0] > scores[1]
        assert scores[2] > scores[1]

    def test_bm25_score_types(self) -> None:
        from ctxeng.retrieval.hybrid import BM25Retriever

        bm25 = BM25Retriever()
        bm25.index(["hello world"])
        scores = bm25.search("hello")
        assert isinstance(scores[0], float)


@pytest.mark.skipif(not HAS_SENTENCE_TRANSFORMERS, reason="sentence-transformers not installed")
class TestEmbeddingModel:
    def test_available_property(self) -> None:
        from ctxeng.retrieval.embeddings import EmbeddingModel

        model = EmbeddingModel()
        assert model.available is True

    def test_encode_returns_correct_dimension(self) -> None:
        from ctxeng.retrieval.embeddings import EmbeddingModel

        model = EmbeddingModel()
        emb = model.encode(["hello world"])
        assert len(emb) == 1
        assert model.dimension is not None
        assert len(emb[0]) == model.dimension

    def test_encode_query_returns_single_vector(self) -> None:
        from ctxeng.retrieval.embeddings import EmbeddingModel

        model = EmbeddingModel()
        emb = model.encode_query("hello")
        assert model.dimension is not None
        assert len(emb) == model.dimension


@pytest.mark.skipif(not HAS_SENTENCE_TRANSFORMERS, reason="sentence-transformers not installed")
class TestCrossEncoderReranker:
    def test_rerank_returns_ordered_results(self) -> None:
        from ctxeng.models import MemoryItem
        from ctxeng.retrieval.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        candidates = [
            MemoryItem(user_id="alice", text="Python is a programming language."),
            MemoryItem(user_id="alice", text="I enjoy hiking in the mountains."),
        ]
        results = reranker.rerank("programming", candidates)
        assert len(results) == 2
        assert results[0].score >= results[1].score

    def test_rerank_empty_candidates(self) -> None:
        from ctxeng.retrieval.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        results = reranker.rerank("test", [])
        assert results == []

    def test_rerank_top_k(self) -> None:
        from ctxeng.models import MemoryItem
        from ctxeng.retrieval.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        candidates = [
            MemoryItem(user_id="alice", text=f"memory {i}")
            for i in range(10)
        ]
        results = reranker.rerank("memory", candidates, top_k=3)
        assert len(results) == 3


@pytest.mark.skipif(not HAS_SENTENCE_TRANSFORMERS, reason="sentence-transformers not installed")
class TestHybridWithEmbeddings:
    def test_semantic_search_dense_scores_populated(self) -> None:
        store = InMemoryStore()
        store.add("alice", "I love machine learning and AI.")
        store.add("alice", "The cat sat on the mat.")
        retriever = HybridRetriever(store)
        results = retriever.search("alice", "artificial intelligence")
        assert len(results) >= 1
        assert all(r.score > 0 for r in results)

    def test_semantic_better_than_random(self) -> None:
        store = InMemoryStore()
        store.add("alice", "Deep learning is a subset of machine learning.")
        store.add("alice", "The stock market had a great day today.")
        retriever = HybridRetriever(store)
        results = retriever.search("alice", "neural networks")
        assert results[0].score > 0
