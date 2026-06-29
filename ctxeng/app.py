from __future__ import annotations

from ctxeng.core.context_manager import ContextManager
from ctxeng.models import ConversationTurn
from ctxeng.stores.memory import InMemoryStore


def run_demo() -> str:
    memory_store = InMemoryStore()
    memory_store.add("demo", "The user likes concise answers.")
    manager = ContextManager(memory_store=memory_store)

    turns = [
        ConversationTurn(role="user", content="Help me plan a trip."),
        ConversationTurn(role="assistant", content="I can help outline a simple itinerary."),
    ]

    return manager.build_prompt("demo", turns, "Summarize the trip plan")


if __name__ == "__main__":
    print(run_demo())
