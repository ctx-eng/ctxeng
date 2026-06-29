from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ctxeng.models import MemoryItem


class ContextStore(ABC):
    @abstractmethod
    def add(self, user_id: str, text: str, metadata: Optional[dict] = None) -> MemoryItem:
        ...

    @abstractmethod
    def search(self, user_id: str, query: str, top_k: int = 10) -> list[MemoryItem]:
        ...

    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        ...

    @abstractmethod
    def list(self, user_id: str) -> list[MemoryItem]:
        ...

    @abstractmethod
    def clear(self) -> None:
        ...
