from __future__ import annotations

import pytest

from ctxeng.assembly.assembler import ContextAssembler
from ctxeng.eval.benchmark import BenchmarkRunner, format_benchmark_result
from ctxeng.eval.datasets import (
    EvalDataset,
    EvalQuery,
    BUILT_IN_DATASETS,
    get_dataset,
    list_datasets,
)
from ctxeng.eval.judge import CorrectnessEvaluator
from ctxeng.eval.metrics import CompressionMetrics, ContextMetrics
from ctxeng.models import MemoryItem
from ctxeng.retrieval.hybrid import HybridRetriever
from ctxeng.stores.memory import InMemoryStore


class TestContextMetrics:
    def test_compression_ratio(self) -> None:
        assert CompressionMetrics.compression_ratio(100, 50) == 0.5
        assert CompressionMetrics.compression_ratio(100, 100) == 0.0
        assert CompressionMetrics.compression_ratio(0, 0) == 0.0

    def test_fact_preservation_full(self) -> None:
        assert CompressionMetrics.fact_preservation_rate({"a", "b"}, {"a", "b"}) == 1.0

    def test_fact_preservation_partial(self) -> None:
        assert CompressionMetrics.fact_preservation_rate({"a", "b", "c"}, {"a", "b"}) == 2 / 3

    def test_fact_preservation_empty_original(self) -> None:
        assert CompressionMetrics.fact_preservation_rate(set(), {"a"}) == 1.0

    def test_information_density(self) -> None:
        assert CompressionMetrics.information_density(5, 100) == 0.05
        assert CompressionMetrics.information_density(0, 100) == 0.0

    def test_correctness_evaluator_no_reference(self) -> None:
        evaluator = CorrectnessEvaluator()
        result = evaluator.evaluate("question", "answer")
        assert result.score == 0.0
        assert "No reference" in result.explanation

    def test_correctness_evaluator_with_reference(self) -> None:
        evaluator = CorrectnessEvaluator()
        result = evaluator.evaluate(
            "What color is the sky?",
            "The sky is blue.",
            "The sky appears blue.",
        )
        assert 0.0 < result.score <= 1.0

    def test_correctness_evaluator_no_overlap(self) -> None:
        evaluator = CorrectnessEvaluator()
        result = evaluator.evaluate(
            "What is 2+2?",
            "blue sky ocean waves",
            "answer equals four exactly",
        )
        assert result.score == 0.0

    def test_memory_retention_dataset_loaded(self) -> None:
        from ctxeng.eval.datasets import BUILT_IN_DATASETS
        assert "memory_retention" in BUILT_IN_DATASETS
        ds = BUILT_IN_DATASETS["memory_retention"]
        assert len(ds.memories) == 7
        assert len(ds.queries) == 7

    def test_memory_retention_benchmark(self) -> None:
        store = InMemoryStore()
        retriever = HybridRetriever(store)
        assembler = ContextAssembler(store=store)
        runner = BenchmarkRunner(store, retriever, assembler)
        result = runner.run_dataset(BUILT_IN_DATASETS["memory_retention"])
        assert result.num_queries == 7

    def test_precision_at_k_exact(self) -> None:
        assert ContextMetrics.precision_at_k(["a", "b"], {"a"}, 1) == 1.0

    def test_precision_at_k_partial(self) -> None:
        assert ContextMetrics.precision_at_k(["a", "b"], {"a"}, 2) == 0.5

    def test_precision_at_k_zero(self) -> None:
        assert ContextMetrics.precision_at_k(["a", "b"], {"c"}, 2) == 0.0

    def test_precision_at_k_empty_retrieved(self) -> None:
        assert ContextMetrics.precision_at_k([], {"a"}, 5) == 0.0

    def test_precision_at_k_k_zero(self) -> None:
        assert ContextMetrics.precision_at_k(["a"], {"a"}, 0) == 0.0

    def test_recall_at_k_full(self) -> None:
        assert ContextMetrics.recall_at_k(["a", "b"], {"a", "b"}, 5) == 1.0

    def test_recall_at_k_partial(self) -> None:
        assert ContextMetrics.recall_at_k(["a"], {"a", "b"}, 5) == 0.5

    def test_recall_at_k_zero(self) -> None:
        assert ContextMetrics.recall_at_k(["a"], {"b"}, 5) == 0.0

    def test_recall_at_k_empty_relevant(self) -> None:
        assert ContextMetrics.recall_at_k(["a"], set(), 5) == 0.0

    def test_reciprocal_rank_first(self) -> None:
        assert ContextMetrics.reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0

    def test_reciprocal_rank_second(self) -> None:
        assert ContextMetrics.reciprocal_rank(["c", "a", "b"], {"a"}) == 0.5

    def test_reciprocal_rank_not_found(self) -> None:
        assert ContextMetrics.reciprocal_rank(["a", "b"], {"c"}) == 0.0

    def test_reciprocal_rank_empty_retrieved(self) -> None:
        assert ContextMetrics.reciprocal_rank([], {"a"}) == 0.0

    def test_mrr_multiple_queries(self) -> None:
        queries = [["a", "b"], ["c", "d", "a"]]
        relevants = [{"a"}, {"a"}]
        result = ContextMetrics.mrr(queries, relevants)
        expected = (1.0 + 1.0 / 3.0) / 2.0
        assert abs(result - expected) < 1e-6

    def test_mrr_empty(self) -> None:
        assert ContextMetrics.mrr([], []) == 0.0

    def test_token_efficiency_half(self) -> None:
        assert ContextMetrics.token_efficiency(50, 100) == 0.5

    def test_token_efficiency_zero_total(self) -> None:
        assert ContextMetrics.token_efficiency(50, 0) == 0.0

    def test_budget_utilization_half(self) -> None:
        assert ContextMetrics.budget_utilization(500, 1000) == 0.5

    def test_budget_utilization_over(self) -> None:
        assert ContextMetrics.budget_utilization(1500, 1000) == 1.0

    def test_budget_utilization_zero_budget(self) -> None:
        assert ContextMetrics.budget_utilization(100, 0) == 0.0

    def test_average_precision(self) -> None:
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "c", "e"}
        ap = ContextMetrics.average_precision(retrieved, relevant)
        expected = (1.0 / 1 + 2.0 / 3 + 3.0 / 5) / 3.0
        assert abs(ap - expected) < 1e-6

    def test_average_precision_no_relevant(self) -> None:
        assert ContextMetrics.average_precision(["a"], set()) == 0.0

    def test_average_precision_no_hits(self) -> None:
        assert ContextMetrics.average_precision(["a", "b"], {"c"}) == 0.0


class TestDatasets:
    def test_built_in_datasets_have_queries(self) -> None:
        for name, ds in BUILT_IN_DATASETS.items():
            assert len(ds.queries) > 0
            assert len(ds.memories) > 0
            assert len(ds.name) > 0

    def test_get_dataset_by_name(self) -> None:
        ds = get_dataset("simple_preferences")
        assert ds.name == "simple_preferences"

    def test_get_dataset_unknown_raises(self) -> None:
        with pytest.raises(KeyError):
            get_dataset("nonexistent")

    def test_list_datasets(self) -> None:
        names = list_datasets()
        assert "simple_preferences" in names
        assert "multi_topic_conversations" in names
        assert "long_term_cross_session" in names

    def test_all_datasets_relevant_ids_exist(self) -> None:
        for ds in BUILT_IN_DATASETS.values():
            memory_ids = {m.id for m in ds.memories}
            for q in ds.queries:
                for rid in q.relevant_ids:
                    assert rid in memory_ids, f"{rid} not in {ds.name} memories"


class TestBenchmarkRunner:
    def test_benchmark_returns_result(self) -> None:
        store = InMemoryStore()
        retriever = HybridRetriever(store)
        assembler = ContextAssembler(store=store)
        runner = BenchmarkRunner(store, retriever, assembler)
        result = runner.run_dataset(BUILT_IN_DATASETS["simple_preferences"])
        assert result.num_queries > 0
        assert result.avg_precision_at_1 >= 0
        assert result.mrr >= 0
        assert result.total_duration_ms > 0

    def test_benchmark_all_datasets(self) -> None:
        store = InMemoryStore()
        retriever = HybridRetriever(store)
        assembler = ContextAssembler(store=store)
        runner = BenchmarkRunner(store, retriever, assembler)
        results = runner.run_all()
        assert len(results) == len(BUILT_IN_DATASETS)
        for name, result in results.items():
            assert result.dataset_name == name
            assert result.num_queries > 0

    def test_benchmark_query_has_latency(self) -> None:
        store = InMemoryStore()
        retriever = HybridRetriever(store)
        assembler = ContextAssembler(store=store)
        runner = BenchmarkRunner(store, retriever, assembler)
        result = runner.run_dataset(BUILT_IN_DATASETS["simple_preferences"])
        for qr in result.queries:
            assert qr.latency_ms > 0

    def test_benchmark_query_precision_in_range(self) -> None:
        store = InMemoryStore()
        retriever = HybridRetriever(store)
        assembler = ContextAssembler(store=store)
        runner = BenchmarkRunner(store, retriever, assembler)
        result = runner.run_dataset(BUILT_IN_DATASETS["simple_preferences"])
        for qr in result.queries:
            assert 0 <= qr.precision_at_1 <= 1

    def test_format_benchmark_result_contains_metrics(self) -> None:
        store = InMemoryStore()
        retriever = HybridRetriever(store)
        assembler = ContextAssembler(store=store)
        runner = BenchmarkRunner(store, retriever, assembler)
        result = runner.run_dataset(BUILT_IN_DATASETS["simple_preferences"])
        formatted = format_benchmark_result(result)
        assert "P@1" in formatted
        assert "MRR" in formatted
        assert result.dataset_name in formatted
