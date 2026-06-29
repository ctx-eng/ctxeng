from __future__ import annotations

from typing import List, Optional

from ctxeng.models import MemoryItem


def _trigram_similarity(a: str, b: str) -> float:
    def trigrams(s: str) -> set:
        return {s[i:i+3] for i in range(len(s) - 2)}
    t1 = trigrams(a.lower())
    t2 = trigrams(b.lower())
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


class Prioritizer:
    def __init__(self, diversity_weight: float = 0.3, dedup_threshold: float = 0.85) -> None:
        self.diversity_weight = diversity_weight
        self.dedup_threshold = dedup_threshold

    def deduplicate(self, items: List[MemoryItem]) -> List[MemoryItem]:
        if len(items) < 2:
            return items

        kept: List[MemoryItem] = []
        for item in items:
            is_dup = False
            for existing in kept:
                if _trigram_similarity(item.text, existing.text) >= self.dedup_threshold:
                    if item.score > existing.score:
                        kept.remove(existing)
                    else:
                        is_dup = True
                    break
            if not is_dup:
                kept.append(item)
        return kept

    def prioritize(self, items: List[MemoryItem], top_k: Optional[int] = None) -> List[MemoryItem]:
        if not items:
            return []

        sorted_items = sorted(items, key=lambda x: x.score, reverse=True)

        if self.diversity_weight <= 0 or len(sorted_items) <= 1:
            return sorted_items[:top_k]

        selected: List[MemoryItem] = [sorted_items[0]]
        candidates = sorted_items[1:]

        while candidates and (top_k is None or len(selected) < top_k):
            mmr_scores = []
            for c in candidates:
                rel = c.score
                div = max(
                    _trigram_similarity(c.text, s.text) for s in selected
                )
                mmr = self.diversity_weight * rel - (1 - self.diversity_weight) * div
                mmr_scores.append(mmr)

            best_idx = mmr_scores.index(max(mmr_scores))
            selected.append(candidates.pop(best_idx))

        return selected

    def format_memories(self, items: List[MemoryItem]) -> str:
        if not items:
            return "- none"
        return "\n".join(f"- {m.text}" for m in items)

    def format_history(self, turns: List, roles: Optional[List[str]] = None) -> str:
        if not turns:
            return "- no prior conversation"
        return "\n".join(f"{t.role}: {t.content}" for t in turns)
