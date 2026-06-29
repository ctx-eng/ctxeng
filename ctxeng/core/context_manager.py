from __future__ import annotations

from typing import List, Optional

from ctxeng.models import ConversationTurn
from ctxeng.stores.base import ContextStore
from ctxeng.stores.memory import InMemoryStore


class ContextManager:
    def __init__(self, memory_store: Optional[ContextStore] = None) -> None:
        self.memory_store = memory_store or InMemoryStore()

    def build_prompt(self, user_id: str, turns: List[ConversationTurn], current_query: str) -> str:
        memories = self.memory_store.search(user_id, current_query)
        memory_block = "\n".join(f"- {memory.text}" for memory in memories) if memories else "- none"

        history_block = "\n".join(
            f"{turn.role}: {turn.content}" for turn in turns
        ) if turns else "- no prior conversation"

        return (
            "You are CtxEng.\n"
            f"Relevant memories:\n{memory_block}\n\n"
            f"Conversation history:\n{history_block}\n\n"
            f"Current request:\n{current_query}\n"
        )
