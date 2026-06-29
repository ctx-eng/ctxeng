from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ctxeng.models import ConversationTurn, MemoryItem
from ctxeng.retrieval.hybrid import HybridRetriever
from ctxeng.routing.diff import ContextDiff
from ctxeng.routing.lifecycle import LifecycleManager
from ctxeng.stores.base import ContextStore


@dataclass
class RouterResult:
    memories: List[MemoryItem]
    diff: ContextDiff
    step: int = 0
    trace: Dict = field(default_factory=dict)


class ContextRouter:
    def __init__(
        self,
        store: ContextStore,
        retriever: Optional[HybridRetriever] = None,
        lifecycle: Optional[LifecycleManager] = None,
        max_history: int = 20,
    ) -> None:
        self.store = store
        self.retriever = retriever or HybridRetriever(store)
        self.lifecycle = lifecycle or LifecycleManager(store)
        self.max_history = max_history
        self._session_steps: Dict[str, int] = {}
        self._previous_context: Dict[str, List[MemoryItem]] = {}

    def step(
        self,
        user_id: str,
        query: str,
        turns: Optional[List[ConversationTurn]] = None,
    ) -> RouterResult:
        if user_id not in self._session_steps:
            self._session_steps[user_id] = 0
            self._previous_context[user_id] = []

        self._session_steps[user_id] += 1
        step_num = self._session_steps[user_id]

        memories = self.retriever.search(user_id, query)

        previous = self._previous_context.get(user_id, [])
        diff = ContextDiff.compute(previous, memories)

        for m in memories:
            self.lifecycle.record_access(m.id)

        self._previous_context[user_id] = memories

        archived = self.lifecycle.archive_stale(user_id)

        return RouterResult(
            memories=memories,
            diff=diff,
            step=step_num,
            trace={
                "step": step_num,
                "user_id": user_id,
                "query": query,
                "memories_found": len(memories),
                "added": len(diff.added),
                "removed": len(diff.removed),
                "archived": archived,
                "has_diff": diff.has_changes,
            },
        )

    def reset_session(self, user_id: str) -> None:
        self._session_steps.pop(user_id, None)
        self._previous_context.pop(user_id, None)

    def get_step_count(self, user_id: str) -> int:
        return self._session_steps.get(user_id, 0)

    def branch(
        self,
        user_id: str,
        query: str,
        base_memories: List[MemoryItem],
    ) -> RouterResult:
        memories = self.retriever.search(user_id, query)
        all_memories = list({m.id: m for m in base_memories + memories}.values())
        diff = ContextDiff.compute(base_memories, all_memories)

        for m in all_memories:
            self.lifecycle.record_access(m.id)

        return RouterResult(
            memories=all_memories,
            diff=diff,
            step=self._session_steps.get(user_id, 0) + 1,
            trace={"branch": True, "query": query, "memories": len(all_memories)},
        )
