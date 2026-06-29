from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from ctxeng.models import MemoryItem


class ContextStore(ABC):
    @abstractmethod
    def add(self, user_id: str, text: str, metadata: Optional[dict] = None) -> MemoryItem:
        ...

    @abstractmethod
    def search(self, user_id: str, query: str, top_k: int = 10) -> List[MemoryItem]:
        ...

    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        ...

    @abstractmethod
    def list(self, user_id: str) -> List[MemoryItem]:
        ...

    @abstractmethod
    def clear(self) -> None:
        ...
