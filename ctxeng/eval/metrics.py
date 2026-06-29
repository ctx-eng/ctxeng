from __future__ import annotations

from typing import List, Set


class ContextMetrics:
    @staticmethod
    def precision_at_k(
        retrieved: List[str],
        relevant: Set[str],
        k: int,
    ) -> float:
        if k <= 0 or not retrieved:
            return 0.0
        top_k = retrieved[:k]
        if not top_k:
            return 0.0
        hits = sum(1 for r in top_k if r in relevant)
        return hits / k

    @staticmethod
    def recall_at_k(
        retrieved: List[str],
        relevant: Set[str],
        k: int,
    ) -> float:
        if not relevant:
            return 0.0
        top_k = retrieved[:k]
        if not top_k:
            return 0.0
        hits = sum(1 for r in top_k if r in relevant)
        return hits / len(relevant)

    @staticmethod
    def reciprocal_rank(
        retrieved: List[str],
        relevant: Set[str],
    ) -> float:
        for i, r in enumerate(retrieved):
            if r in relevant:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def mrr(
        queries: List[List[str]],
        relevants: List[Set[str]],
    ) -> float:
        if not queries:
            return 0.0
        total = sum(
            ContextMetrics.reciprocal_rank(q, rel)
            for q, rel in zip(queries, relevants)
        )
        return total / len(queries)

    @staticmethod
    def token_efficiency(
        relevant_tokens: int,
        total_prompt_tokens: int,
    ) -> float:
        if total_prompt_tokens <= 0:
            return 0.0
        return relevant_tokens / total_prompt_tokens

    @staticmethod
    def budget_utilization(
        actual_tokens: int,
        budget: int,
    ) -> float:
        if budget <= 0:
            return 0.0
        return min(1.0, actual_tokens / budget)

    @staticmethod
    def average_precision(
        retrieved: List[str],
        relevant: Set[str],
    ) -> float:
        if not relevant:
            return 0.0
        hits = 0
        sum_precision = 0.0
        for i, r in enumerate(retrieved):
            if r in relevant:
                hits += 1
                sum_precision += hits / (i + 1)
        if hits == 0:
            return 0.0
        return sum_precision / len(relevant)
