from __future__ import annotations

from ctxeng.core.context_manager import ContextManager
from ctxeng.core.profile import ProfileStore


def test_profile_store_get_or_create() -> None:
    ps = ProfileStore()
    profile = ps.get_or_create("u1")
    assert profile.user_id == "u1"
    assert profile.name == ""

    same = ps.get_or_create("u1", "Alice")
    assert same.name == ""


def test_profile_get_nonexistent() -> None:
    ps = ProfileStore()
    assert ps.get("nonexistent") is None


def test_set_and_get_preference() -> None:
    ps = ProfileStore()
    pref = ps.set_preference("u1", "language", "python", "tech")
    assert pref.key == "language"
    assert pref.value == "python"

    retrieved = ps.get_preference("u1", "language")
    assert retrieved is not None
    assert retrieved.value == "python"

    assert ps.get_preference("u1", "nonexistent") is None


def test_set_preference_overwrites() -> None:
    ps = ProfileStore()
    ps.set_preference("u1", "lang", "python")
    ps.set_preference("u1", "lang", "rust")
    retrieved = ps.get_preference("u1", "lang")
    assert retrieved is not None
    assert retrieved.value == "rust"


def test_set_tags() -> None:
    ps = ProfileStore()
    ps.set_tags("u1", ["beginner", "power-user"])
    profile = ps.get("u1")
    assert profile is not None
    assert "beginner" in profile.tags
    assert "power-user" in profile.tags


def test_to_context_no_profile() -> None:
    ps = ProfileStore()
    ctx = ps.to_context("nonexistent")
    assert ctx == "- no profile"


def test_to_context_with_preferences() -> None:
    ps = ProfileStore()
    ps.set_preference("u1", "language", "python", "tech")
    ps.set_preference("u1", "theme", "dark", "ui")
    ps.set_tags("u1", ["developer"])
    ctx = ps.to_context("u1")
    assert "language:" in ctx
    assert "python" in ctx
    assert "developer" in ctx
    assert "Tags:" in ctx


def test_assemble_with_profile() -> None:
    from ctxeng.assembly.assembler import ContextAssembler
    from ctxeng.models import ConversationTurn
    from ctxeng.stores.memory import InMemoryStore

    store = InMemoryStore()
    assembler = ContextAssembler(store=store)
    turns = [ConversationTurn(role="user", content="hi")]
    profile_ctx = "Name: Alice\nPreferences:\n  language: python"

    prompt = assembler.assemble("u1", turns, "what is my language?", profile_context=profile_ctx)
    assert "User profile:" in prompt
    assert "Alice" in prompt
    assert "python" in prompt


def test_context_manager_profile_in_prompt() -> None:
    mgr = ContextManager()
    mgr._profile_store.set_preference("u1", "language", "python")
    mgr._profile_store.set_tags("u1", ["developer"])

    from ctxeng.models import ConversationTurn

    turns = [ConversationTurn(role="user", content="hi")]
    prompt = mgr.build_prompt("u1", turns, "what do I like?")
    assert "User profile:" in prompt
    assert "python" in prompt
