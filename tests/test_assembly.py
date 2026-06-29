from __future__ import annotations

import pytest

from ctxeng.assembly.assembler import ContextAssembler, _estimate_tokens, _truncate_text
from ctxeng.assembly.prioritizer import Prioritizer, _trigram_similarity
from ctxeng.assembly.templates import (
    TEMPLATE_REGISTRY,
    PromptTemplate,
    get_template,
    register_template,
)
from ctxeng.models import ConversationTurn, MemoryItem
from ctxeng.stores.memory import InMemoryStore

try:
    import jinja2  # noqa: F401

    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False


class TestPromptTemplate:
    def test_default_template_exists(self) -> None:
        t = get_template("default")
        assert t is not None
        assert t.name == "default"

    def test_template_slots(self) -> None:
        t = PromptTemplate("Hello {name}, you are {role}.")
        assert set(t.slots) == {"name", "role"}

    def test_render_with_format(self) -> None:
        t = PromptTemplate("Hello {name}!")
        assert t.render(name="World") == "Hello World!"

    def test_render_missing_slot_raises(self) -> None:
        t = PromptTemplate("Hello {name}!")
        with pytest.raises(KeyError):
            t.render(wrong="thing")

    def test_register_and_get(self) -> None:
        register_template("test_tpl", "Custom: {query}")
        t = get_template("test_tpl")
        assert t is not None
        assert t.render(query="hello") == "Custom: hello"

    def test_registry_contains_default(self) -> None:
        assert "default" in TEMPLATE_REGISTRY

    def test_default_template_format(self) -> None:
        t = get_template("default")
        assert t is not None
        result = t.render(
            profile="- no profile", memories="- foo", history="- bar",
            tool_outputs="- none", query="test",
        )
        assert "Relevant memories:" in result
        assert "- foo" in result
        assert "Conversation history:" in result
        assert "- bar" in result
        assert "Current request:" in result
        assert "test" in result


@pytest.mark.skipif(not HAS_JINJA2, reason="jinja2 not installed")
class TestPromptTemplateJinja:
    def test_jinja_render(self) -> None:
        t = PromptTemplate("Hello {{ name }}!")
        assert t.render(name="World") == "Hello World!"

    def test_jinja_with_conditionals(self) -> None:
        t = PromptTemplate("{% if show %}visible{% else %}hidden{% endif %}")
        assert t.render(show=True) == "visible"
        assert t.render(show=False) == "hidden"


class TestTrigramSimilarity:
    def test_identical_strings(self) -> None:
        assert _trigram_similarity("hello world", "hello world") == 1.0

    def test_completely_different(self) -> None:
        assert _trigram_similarity("abc", "xyz") == 0.0

    def test_partial_overlap(self) -> None:
        sim = _trigram_similarity("hello world", "hello there")
        assert 0.0 < sim < 1.0

    def test_short_strings(self) -> None:
        assert _trigram_similarity("a", "b") == 0.0
        assert _trigram_similarity("ab", "ab") == 0.0

    def test_case_insensitive(self) -> None:
        assert _trigram_similarity("Hello World", "hello world") == 1.0


class TestPrioritizer:
    def test_deduplicate_removes_near_duplicates(self) -> None:
        p = Prioritizer(dedup_threshold=0.5)
        items = [
            MemoryItem(user_id="alice", text="Alice likes concise answers.", score=1.0),
            MemoryItem(user_id="alice", text="Alice likes concise answers!!", score=0.5),
        ]
        result = p.deduplicate(items)
        assert len(result) == 1
        assert result[0].score == 1.0

    def test_deduplicate_keeps_different(self) -> None:
        p = Prioritizer(dedup_threshold=0.9)
        items = [
            MemoryItem(user_id="alice", text="Alice likes concise answers.", score=1.0),
            MemoryItem(user_id="alice", text="Bob prefers detailed explanations.", score=0.8),
        ]
        result = p.deduplicate(items)
        assert len(result) == 2

    def test_deduplicate_empty_list(self) -> None:
        p = Prioritizer()
        assert p.deduplicate([]) == []

    def test_deduplicate_single_item(self) -> None:
        p = Prioritizer()
        items = [MemoryItem(user_id="alice", text="hello")]
        assert len(p.deduplicate(items)) == 1

    def test_prioritize_sorts_by_score(self) -> None:
        p = Prioritizer(diversity_weight=0)
        items = [
            MemoryItem(user_id="alice", text="low", score=0.1),
            MemoryItem(user_id="alice", text="high", score=0.9),
            MemoryItem(user_id="alice", text="mid", score=0.5),
        ]
        result = p.prioritize(items)
        assert [r.score for r in result] == [0.9, 0.5, 0.1]

    def test_prioritize_top_k(self) -> None:
        p = Prioritizer(diversity_weight=0)
        items = [MemoryItem(user_id="alice", text=str(i), score=i) for i in range(10)]
        result = p.prioritize(items, top_k=3)
        assert len(result) == 3
        assert [r.score for r in result] == [9, 8, 7]

    def test_prioritize_empty(self) -> None:
        p = Prioritizer()
        assert p.prioritize([]) == []

    def test_format_memories_empty(self) -> None:
        p = Prioritizer()
        assert p.format_memories([]) == "- none"

    def test_format_memories_with_items(self) -> None:
        p = Prioritizer()
        items = [
            MemoryItem(user_id="alice", text="memory 1"),
            MemoryItem(user_id="alice", text="memory 2"),
        ]
        result = p.format_memories(items)
        assert "- memory 1" in result
        assert "- memory 2" in result

    def test_format_history_empty(self) -> None:
        p = Prioritizer()
        assert p.format_history([]) == "- no prior conversation"

    def test_format_history_with_turns(self) -> None:
        p = Prioritizer()
        turns = [
            ConversationTurn(role="user", content="hello"),
            ConversationTurn(role="assistant", content="hi there"),
        ]
        result = p.format_history(turns)
        assert "user: hello" in result
        assert "assistant: hi there" in result


class TestTokenEstimation:
    def test_estimate_empty(self) -> None:
        assert _estimate_tokens("") == 1

    def test_estimate_non_empty(self) -> None:
        assert _estimate_tokens("hello world") > 0

    def test_token_ratio_approx(self) -> None:
        short = _estimate_tokens("a" * 100)
        long = _estimate_tokens("a" * 1000)
        # longer text should have more tokens
        assert long > short


class TestTruncateText:
    def test_short_text_not_truncated(self) -> None:
        assert _truncate_text("hello", 100) == "hello"

    def test_truncation_reduces_length(self) -> None:
        text = "word " * 500
        truncated = _truncate_text(text, 10)
        assert len(truncated) < len(text)


class TestContextAssembler:
    def test_assemble_basic(self) -> None:
        store = InMemoryStore()
        store.add("alice", "Alice likes concise answers.")
        assembler = ContextAssembler(store=store, max_tokens=4096)
        result = assembler.assemble(
            "alice",
            [ConversationTurn(role="user", content="Hello?")],
            "What is the weather?",
        )
        assert "Alice likes concise answers." in result
        assert "Hello?" in result
        assert "What is the weather?" in result

    def test_assemble_no_memories(self) -> None:
        store = InMemoryStore()
        assembler = ContextAssembler(store=store)
        result = assembler.assemble("alice", [], "test")
        assert "- none" in result

    def test_assemble_no_history(self) -> None:
        store = InMemoryStore()
        assembler = ContextAssembler(store=store)
        result = assembler.assemble("alice", [], "test")
        assert "- no prior conversation" in result

    def test_assemble_respects_token_budget(self) -> None:
        store = InMemoryStore()
        for i in range(100):
            store.add("alice", f"memory number {i} with some extra text to fill tokens")
        assembler = ContextAssembler(store=store, max_tokens=100)
        result = assembler.assemble(
            "alice",
            [ConversationTurn(role="user", content="hello")],
            "test",
        )
        assert _estimate_tokens(result) <= 150  # allow small overhead

    def test_assemble_with_very_small_budget(self) -> None:
        store = InMemoryStore()
        store.add("alice", "A" * 1000)
        assembler = ContextAssembler(store=store, max_tokens=10)
        result = assembler.assemble("alice", [], "test")
        assert len(result) > 0

    def test_assemble_unknown_user_returns_empty_memories(self) -> None:
        store = InMemoryStore()
        store.add("alice", "Alice's memory")
        assembler = ContextAssembler(store=store)
        result = assembler.assemble("bob", [], "test")
        assert "- none" in result

    def test_assemble_deduplicates(self) -> None:
        store = InMemoryStore()
        store.add("alice", "Alice likes concise answers.")
        store.add("alice", "Alice likes concise answers!!")
        store.add("alice", "Alice likes concise answers!!!")
        assembler = ContextAssembler(store=store, max_tokens=4096)
        result = assembler.assemble("alice", [], "concise")
        # dedup should collapse near-duplicates
        count = result.count("- Alice likes concise answers")
        assert count <= 2  # at most 2 after dedup (could be 1 or 2)

    def test_assemble_prioritizes_by_score(self) -> None:
        store = InMemoryStore()
        store.add("alice", "low relevance text")
        store.add("alice", "highly relevant content about the query subject")
        assembler = ContextAssembler(store=store)
        result = assembler.assemble("alice", [], "query subject")
        lines = [line for line in result.split("\n") if line.startswith("- ")]
        assert len(lines) >= 1
