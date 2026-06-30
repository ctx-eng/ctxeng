from __future__ import annotations

from collections import defaultdict

from ctxeng.compression.summarizer import ContextSummarizer
from ctxeng.models import MemoryItem
from ctxeng.stores.base import ContextStore


class MemoryConsolidator:
    def __init__(
        self,
        store: ContextStore,
        turn_threshold: int = 10,
        batch_size: int = 5,
        session_threshold: int = 5,
        summarizer: ContextSummarizer | None = None,
    ) -> None:
        self.store = store
        self.turn_threshold = turn_threshold
        self.batch_size = batch_size
        self.session_threshold = session_threshold
        self.summarizer = summarizer or ContextSummarizer()
        self._turn_count: dict[str, int] = defaultdict(int)
        self._session_count: dict[str, int] = defaultdict(int)

    def record_turn(self, user_id: str, text: str) -> MemoryItem:
        memory = self.store.add(user_id, text, metadata={"type": "working"})
        self._turn_count[user_id] += 1

        if self._turn_count[user_id] >= self.turn_threshold:
            self._consolidate(user_id)

        return memory

    def _consolidate(self, user_id: str) -> None:
        all_memories = self.store.list(user_id)
        working = [m for m in all_memories if m.metadata.get("type") == "working"]

        if len(working) < self.batch_size:
            return

        to_consolidate = working[: self.batch_size]
        remaining = working[self.batch_size :]

        texts = [m.text for m in to_consolidate]
        combined = " ".join(texts)
        summary_text = self.summarizer.summarize(combined)

        if summary_text:
            self.store.add(
                user_id,
                summary_text,
                metadata={
                    "type": "episodic",
                    "consolidated_from": [m.id for m in to_consolidate],
                },
            )

        for m in to_consolidate:
            self.store.delete(m.id)

        for m in remaining:
            self.store.delete(m.id)
            self.store.add(user_id, m.text, metadata={"type": "working"})

        self._turn_count[user_id] = len(remaining)

        self._session_count[user_id] += 1
        if self._session_count[user_id] >= self.session_threshold:
            self._consolidate_to_long_term(user_id)

    def _consolidate_to_long_term(self, user_id: str) -> None:
        all_memories = self.store.list(user_id)
        episodic = [m for m in all_memories if m.metadata.get("type") == "episodic"]

        if not episodic:
            return

        texts = [m.text for m in episodic]
        combined = "\n".join(texts)
        summary_text = self.summarizer.summarize(combined)

        if summary_text:
            self.store.add(
                user_id,
                summary_text,
                metadata={
                    "type": "long_term",
                    "consolidated_from": [m.id for m in episodic],
                },
            )

        for m in episodic:
            self.store.delete(m.id)

        self._session_count[user_id] = 0

    def get_memory_summary(self, user_id: str) -> str:
        memories = self.store.list(user_id)
        long_term = [m for m in memories if m.metadata.get("type") == "long_term"]
        episodic = [m for m in memories if m.metadata.get("type") == "episodic"]
        working = [
            m for m in memories if m.metadata.get("type") == "working" or "type" not in m.metadata
        ]

        parts: list[str] = []
        if long_term:
            parts.append("Long-term memory:")
            for m in long_term:
                parts.append(f"- {m.text}")

        if episodic:
            parts.append("\nPast conversations:")
            for m in episodic:
                parts.append(f"- {m.text}")

        if working:
            parts.append("\nRecent messages:")
            for m in working:
                parts.append(f"- {m.text}")

        return "\n".join(parts) if parts else "No memories yet."
