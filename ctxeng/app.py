from __future__ import annotations

from ctxeng.core.context_manager import ContextManager
from ctxeng.core.memory_store import MemoryStore
from ctxeng.models import ConversationTurn


def run_demo() -> str:
    memory_store = MemoryStore()
    memory_store.add_memory("demo", "The user likes concise answers.")
    manager = ContextManager(memory_store=memory_store)

    turns = [
        ConversationTurn(role="user", content="Help me plan a trip."),
        ConversationTurn(role="assistant", content="I can help outline a simple itinerary."),
    ]

    return manager.build_prompt("demo", turns, "Summarize the trip plan")


if __name__ == "__main__":
    print(run_demo())
