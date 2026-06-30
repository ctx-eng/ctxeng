from __future__ import annotations

from ctxeng.models import MemoryItem
from ctxeng.stores.base import ContextStore


class InMemoryStore(ContextStore):
    def __init__(self) -> None:
        self._memories: list[MemoryItem] = []

    def add(self, user_id: str, text: str, metadata: dict | None = None) -> MemoryItem:
        memory = MemoryItem(
            user_id=user_id,
            text=text,
            metadata=metadata or {},
        )
        self._memories.append(memory)
        return memory

    def search(self, user_id: str, query: str, top_k: int = 10) -> list[MemoryItem]:
        query_lower = query.lower()
        matches = [
            m for m in self._memories if m.user_id == user_id and query_lower in m.text.lower()
        ]
        if matches:
            for m in matches:
                m.score = 1.0
            return matches[:top_k]

        results = [m for m in self._memories if m.user_id == user_id]
        for m in results:
            m.score = 0.5
        return results[:top_k]

    def delete(self, memory_id: str) -> bool:
        for i, m in enumerate(self._memories):
            if m.id == memory_id:
                self._memories.pop(i)
                return True
        return False

    def list(self, user_id: str) -> list[MemoryItem]:
        return [m for m in self._memories if m.user_id == user_id]

    def clear(self) -> None:
        self._memories.clear()
