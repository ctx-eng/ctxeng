from __future__ import annotations

import time
from dataclasses import dataclass, field

from ctxeng.assembly.assembler import ContextAssembler, _estimate_tokens
from ctxeng.eval.datasets import BUILT_IN_DATASETS, EvalDataset
from ctxeng.eval.metrics import ContextMetrics
from ctxeng.retrieval.hybrid import HybridRetriever
from ctxeng.stores.base import ContextStore


@dataclass
class QueryResult:
    query: str
    retrieved_ids: list[str]
    relevant_ids: set[str]
    latency_ms: float
    precision_at_1: float
    precision_at_3: float
    precision_at_5: float
    recall_at_3: float
    recall_at_5: float
    reciprocal_rank: float
    average_precision: float
    token_efficiency: float
    total_tokens: int


@dataclass
class BenchmarkResult:
    dataset_name: str
    num_queries: int
    avg_precision_at_1: float
    avg_precision_at_3: float
    avg_precision_at_5: float
    avg_recall_at_3: float
    avg_recall_at_5: float
    mrr: float
    map_score: float
    avg_token_efficiency: float
    avg_latency_ms: float
    total_duration_ms: float
    queries: list[QueryResult] = field(default_factory=list)


class BenchmarkRunner:
    def __init__(
        self,
        store: ContextStore,
        retriever: HybridRetriever,
        assembler: ContextAssembler,
    ) -> None:
        self.store = store
        self.retriever = retriever
        self.assembler = assembler

    def run_dataset(self, dataset: EvalDataset) -> BenchmarkResult:
        self.store.clear()
        for m in dataset.memories:
            self.store.add(m.user_id, m.text)

        query_results: list[QueryResult] = []
        for eq in dataset.queries:
            t0 = time.perf_counter()
            retrieved = self.retriever.search(eq.user_id, eq.query, top_k=10)
            latency = (time.perf_counter() - t0) * 1000

            retrieved_ids = [m.id for m in retrieved]

            qr = QueryResult(
                query=eq.query,
                retrieved_ids=retrieved_ids,
                relevant_ids=eq.relevant_ids,
                latency_ms=round(latency, 2),
                precision_at_1=ContextMetrics.precision_at_k(retrieved_ids, eq.relevant_ids, 1),
                precision_at_3=ContextMetrics.precision_at_k(retrieved_ids, eq.relevant_ids, 3),
                precision_at_5=ContextMetrics.precision_at_k(retrieved_ids, eq.relevant_ids, 5),
                recall_at_3=ContextMetrics.recall_at_k(retrieved_ids, eq.relevant_ids, 3),
                recall_at_5=ContextMetrics.recall_at_k(retrieved_ids, eq.relevant_ids, 5),
                reciprocal_rank=ContextMetrics.reciprocal_rank(retrieved_ids, eq.relevant_ids),
                average_precision=ContextMetrics.average_precision(retrieved_ids, eq.relevant_ids),
                token_efficiency=0.0,
                total_tokens=0,
            )

            try:
                prompt, _ = (
                    self.assembler.assemble(eq.user_id, [], eq.query)
                    if hasattr(self.assembler, "assemble")
                    else (self.assembler.build_prompt(eq.user_id, [], eq.query), None)  # type: ignore
                )
                total_tok = _estimate_tokens(prompt)
                relevant_tok = sum(
                    _estimate_tokens(m.text)
                    for m in retrieved
                    if m.id in eq.relevant_ids
                )
                qr.token_efficiency = round(ContextMetrics.token_efficiency(relevant_tok, total_tok), 4)
                qr.total_tokens = total_tok
            except Exception:
                pass

            query_results.append(qr)

        all_retrieved = [q.retrieved_ids for q in query_results]
        all_relevant = [q.relevant_ids for q in query_results]

        return BenchmarkResult(
            dataset_name=dataset.name,
            num_queries=len(query_results),
            avg_precision_at_1=round(sum(q.precision_at_1 for q in query_results) / len(query_results), 4),
            avg_precision_at_3=round(sum(q.precision_at_3 for q in query_results) / len(query_results), 4),
            avg_precision_at_5=round(sum(q.precision_at_5 for q in query_results) / len(query_results), 4),
            avg_recall_at_3=round(sum(q.recall_at_3 for q in query_results) / len(query_results), 4),
            avg_recall_at_5=round(sum(q.recall_at_5 for q in query_results) / len(query_results), 4),
            mrr=round(ContextMetrics.mrr(all_retrieved, all_relevant), 4),
            map_score=round(sum(q.average_precision for q in query_results) / len(query_results), 4),
            avg_token_efficiency=round(sum(q.token_efficiency for q in query_results) / len(query_results), 4),
            avg_latency_ms=round(sum(q.latency_ms for q in query_results) / len(query_results), 2),
            total_duration_ms=round(sum(q.latency_ms for q in query_results), 2),
            queries=query_results,
        )

    def run_all(self) -> dict[str, BenchmarkResult]:
        results: dict[str, BenchmarkResult] = {}
        for name, dataset in BUILT_IN_DATASETS.items():
            results[name] = self.run_dataset(dataset)
        return results


def format_benchmark_result(result: BenchmarkResult) -> str:
    lines = [f"Dataset: {result.dataset_name} ({result.num_queries} queries)"]
    lines.append(f"  P@1: {result.avg_precision_at_1:.4f}")
    lines.append(f"  P@3: {result.avg_precision_at_3:.4f}")
    lines.append(f"  P@5: {result.avg_precision_at_5:.4f}")
    lines.append(f"  R@3: {result.avg_recall_at_3:.4f}")
    lines.append(f"  R@5: {result.avg_recall_at_5:.4f}")
    lines.append(f"  MRR: {result.mrr:.4f}")
    lines.append(f"  MAP: {result.map_score:.4f}")
    lines.append(f"  Token efficiency: {result.avg_token_efficiency:.4f}")
    lines.append(f"  Avg latency: {result.avg_latency_ms:.1f}ms")
    lines.append(f"  Total: {result.total_duration_ms:.0f}ms")
    return "\n".join(lines)
