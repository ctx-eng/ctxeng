from ctxeng.core.context_manager import ContextManager
from ctxeng.models import ConversationTurn
from ctxeng.stores.memory import InMemoryStore


def test_memory_store_persists_and_searches():
    store = InMemoryStore()
    store.add("user", "I prefer concise answers.")

    results = store.search("user", "concise")

    assert len(results) == 1
    assert results[0].text == "I prefer concise answers."


def test_context_manager_builds_prompt_with_memory_and_history():
    store = InMemoryStore()
    store.add("user", "I prefer concise answers.")
    manager = ContextManager(memory_store=store)

    turns = [
        ConversationTurn(role="user", content="What is the weather?"),
        ConversationTurn(role="assistant", content="It is sunny today."),
    ]

    prompt = manager.build_prompt("user", turns, "Summarize the plan")

    assert "I prefer concise answers." in prompt
    assert "What is the weather?" in prompt
    assert "Summarize the plan" in prompt
