from __future__ import annotations

from typing import List

from ctxeng.models import MemoryItem


class MemoryStore:
    def __init__(self) -> None:
        self._memories: List[MemoryItem] = []

    def add_memory(self, user_id: str, text: str) -> MemoryItem:
        memory = MemoryItem(text=text, user_id=user_id)
        self._memories.append(memory)
        return memory

    def search(self, user_id: str, query: str) -> List[MemoryItem]:
        query_lower = query.lower()
        matches = [
            memory
            for memory in self._memories
            if memory.user_id == user_id and query_lower in memory.text.lower()
        ]

        if matches:
            return matches

        return [memory for memory in self._memories if memory.user_id == user_id]
