from __future__ import annotations

from typing import List, Optional

from ctxeng.assembly.assembler import ContextAssembler
from ctxeng.models import ConversationTurn
from ctxeng.stores.base import ContextStore
from ctxeng.stores.memory import InMemoryStore


class ContextManager:
    def __init__(
        self,
        memory_store: Optional[ContextStore] = None,
        assembler: Optional[ContextAssembler] = None,
        max_tokens: int = 4096,
    ) -> None:
        self.memory_store = memory_store or InMemoryStore()
        self._assembler = assembler or ContextAssembler(
            store=self.memory_store, max_tokens=max_tokens
        )

    def build_prompt(self, user_id: str, turns: List[ConversationTurn], current_query: str) -> str:
        return self._assembler.assemble(user_id, turns, current_query)
