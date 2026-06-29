from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ctxeng.models import MemoryItem


@dataclass
class ContextDiff:
    added: List[MemoryItem] = field(default_factory=list)
    removed: List[MemoryItem] = field(default_factory=list)
    score_changes: Dict[str, float] = field(default_factory=dict)

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.removed or self.score_changes)

    @staticmethod
    def compute(
        previous: List[MemoryItem],
        current: List[MemoryItem],
    ) -> ContextDiff:
        prev_map: Dict[str, MemoryItem] = {m.id: m for m in previous}
        curr_map: Dict[str, MemoryItem] = {m.id: m for m in current}

        prev_ids = set(prev_map.keys())
        curr_ids = set(curr_map.keys())

        added_ids = curr_ids - prev_ids
        removed_ids = prev_ids - curr_ids
        common_ids = curr_ids & prev_ids

        added = [curr_map[i] for i in added_ids]
        removed = [prev_map[i] for i in removed_ids]

        score_changes: Dict[str, float] = {}
        for mid in common_ids:
            old_score = prev_map[mid].score
            new_score = curr_map[mid].score
            if abs(old_score - new_score) > 1e-6:
                score_changes[mid] = new_score - old_score

        return ContextDiff(
            added=added,
            removed=removed,
            score_changes=score_changes,
        )

    def summary(self, max_items: int = 3) -> str:
        parts: List[str] = []
        if self.added:
            added_texts = [f"+ {m.text[:50]}" for m in self.added[:max_items]]
            parts.append(f"Added ({len(self.added)}): " + "; ".join(added_texts))
            if len(self.added) > max_items:
                parts[-1] += f" (+{len(self.added) - max_items} more)"
        if self.removed:
            removed_texts = [f"- {m.text[:50]}" for m in self.removed[:max_items]]
            parts.append(f"Removed ({len(self.removed)}): " + "; ".join(removed_texts))
            if len(self.removed) > max_items:
                parts[-1] += f" (+{len(self.removed) - max_items} more)"
        if self.score_changes:
            changes = [
                f"id={mid} {delta:+.2f}"
                for mid, delta in list(self.score_changes.items())[:max_items]
            ]
            parts.append(f"Score changes ({len(self.score_changes)}): " + ", ".join(changes))
        return "\n".join(parts) if parts else "No changes"
