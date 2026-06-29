from __future__ import annotations

import warnings
from typing import List, Optional

from ctxeng.models import MemoryItem
from ctxeng.stores.memory import InMemoryStore


class MemoryStore(InMemoryStore):
    def __init__(self) -> None:
        super().__init__()
        warnings.warn(
            "MemoryStore is deprecated, use InMemoryStore from ctxeng.stores.memory instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    def add_memory(self, user_id: str, text: str) -> MemoryItem:
        return self.add(user_id, text)
