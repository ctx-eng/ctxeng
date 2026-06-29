from __future__ import annotations

from ctxeng.compression.consolidator import MemoryConsolidator
from ctxeng.compression.summarizer import (
    ContextSummarizer,
    _split_sentences,
    _tf_scores,
)
from ctxeng.stores.memory import InMemoryStore


class TestSplitSentences:
    def test_basic_splitting(self) -> None:
        result = _split_sentences("Hello world. How are you? I am fine.")
        assert result == ["Hello world.", "How are you?", "I am fine."]

    def test_single_sentence(self) -> None:
        assert _split_sentences("Hello world.") == ["Hello world."]

    def test_empty_text(self) -> None:
        assert _split_sentences("") == []

    def test_no_period(self) -> None:
        assert _split_sentences("hello") == ["hello"]

    def test_multiple_spaces(self) -> None:
        result = _split_sentences("A.  B.   C.")
        assert result == ["A.", "B.", "C."]


class TestTFScores:
    def test_returns_correct_length(self) -> None:
        scores = _tf_scores(["hello world", "foo bar"])
        assert len(scores) == 2

    def test_empty_sentences(self) -> None:
        assert _tf_scores([]) == []

    def test_identical_sentences_differ_by_position_bonus(self) -> None:
        scores = _tf_scores(["hello world", "hello world"])
        expected_diff = 1.0 - 0.5  # position bonus: 1/1 vs 1/2
        assert abs((scores[0] - scores[1]) - expected_diff) < 1e-6

    def test_different_sentences(self) -> None:
        scores = _tf_scores(["rareword", "common common common common common"])
        assert scores[1] >= scores[0]


class TestContextSummarizer:
    def test_short_text_not_summarized(self) -> None:
        s = ContextSummarizer(max_sentences=3)
        assert s.summarize("Hello world.") == "Hello world."

    def test_long_text_is_summarized(self) -> None:
        s = ContextSummarizer(max_sentences=2)
        text = "First sentence about apples. " * 3
        text += "Second sentence about oranges. " * 3
        text += "Third sentence about bananas. " * 3
        result = s.summarize(text)
        sentences = _split_sentences(result)
        assert len(sentences) <= 2

    def test_empty_text(self) -> None:
        s = ContextSummarizer()
        assert s.summarize("") == ""

    def test_whitespace_only(self) -> None:
        s = ContextSummarizer()
        assert s.summarize("   ") == ""

    def test_summarize_fn_called_when_provided(self) -> None:
        def fake_summary(text: str) -> str:
            return "custom summary"
        s = ContextSummarizer(summarize_fn=fake_summary)
        assert s.summarize("long text " * 100) == "custom summary"


class TestSlidingWindow:
    def test_fewer_turns_than_window(self) -> None:
        s = ContextSummarizer()
        turns = ["turn1", "turn2"]
        result = s.sliding_window(turns, window=5)
        assert "turn1" in result
        assert "turn2" in result

    def test_more_turns_than_window(self) -> None:
        s = ContextSummarizer(max_sentences=1)
        turns = ["old context A", "old context B", "recent1", "recent2", "recent3"]
        result = s.sliding_window(turns, window=3)
        assert "recent1" in result
        assert "recent2" in result
        assert "recent3" in result
        assert "[summary]" in result

    def test_exact_window(self) -> None:
        s = ContextSummarizer()
        turns = ["a", "b", "c"]
        result = s.sliding_window(turns, window=3)
        assert "[summary]" not in result
        assert "a" in result

    def test_empty_turns(self) -> None:
        s = ContextSummarizer()
        assert s.sliding_window([]) == ""

    def test_custom_window_from_init(self) -> None:
        s = ContextSummarizer(keep_recent=2)
        turns = ["old", "recent1", "recent2"]
        result = s.sliding_window(turns)
        assert "recent1" in result
        assert "recent2" in result
        assert "[summary]" in result


class TestMemoryConsolidator:
    def test_record_turn_adds_working_memory(self) -> None:
        store = InMemoryStore()
        consolidator = MemoryConsolidator(store, turn_threshold=20)
        memory = consolidator.record_turn("alice", "Hello world")
        assert memory.text == "Hello world"
        assert memory.metadata.get("type") == "working"

    def test_consolidation_triggers_at_threshold(self) -> None:
        store = InMemoryStore()
        consolidator = MemoryConsolidator(
            store, turn_threshold=5, batch_size=3
        )
        for i in range(5):
            consolidator.record_turn("alice", f"turn {i}")

        memories = store.list("alice")
        types = [m.metadata.get("type") for m in memories]
        assert "episodic" in types

    def test_consolidation_removes_working_memories(self) -> None:
        store = InMemoryStore()
        consolidator = MemoryConsolidator(
            store, turn_threshold=5, batch_size=3
        )
        for i in range(5):
            consolidator.record_turn("alice", f"turn {i}")

        working = [m for m in store.list("alice") if m.metadata.get("type") == "working"]
        assert len(working) <= 2

    def test_consolidation_produces_episodic_summary(self) -> None:
        store = InMemoryStore()
        consolidator = MemoryConsolidator(
            store, turn_threshold=3, batch_size=2
        )
        consolidator.record_turn("alice", "I love programming in Python.")
        consolidator.record_turn("alice", "Python is great for data science.")
        consolidator.record_turn("alice", "I also enjoy machine learning.")

        episodic = [
            m for m in store.list("alice")
            if m.metadata.get("type") == "episodic"
        ]
        assert len(episodic) >= 1
        assert len(episodic[0].text) > 0

    def test_no_consolidation_below_threshold(self) -> None:
        store = InMemoryStore()
        consolidator = MemoryConsolidator(store, turn_threshold=10, batch_size=3)
        for i in range(3):
            consolidator.record_turn("alice", f"turn {i}")

        episodic = [
            m for m in store.list("alice")
            if m.metadata.get("type") == "episodic"
        ]
        assert len(episodic) == 0

    def test_get_memory_summary_returns_formatted(self) -> None:
        store = InMemoryStore()
        consolidator = MemoryConsolidator(store, turn_threshold=10)
        consolidator.record_turn("alice", "test memory")
        summary = consolidator.get_memory_summary("alice")
        assert "test memory" in summary

    def test_get_memory_summary_empty(self) -> None:
        store = InMemoryStore()
        consolidator = MemoryConsolidator(store)
        assert "No memories yet." in consolidator.get_memory_summary("alice")

    def test_multiple_consolidations(self) -> None:
        store = InMemoryStore()
        consolidator = MemoryConsolidator(
            store, turn_threshold=4, batch_size=2
        )
        for i in range(8):
            consolidator.record_turn("alice", f"turn {i}")

        episodic = [
            m for m in store.list("alice")
            if m.metadata.get("type") == "episodic"
        ]
        assert len(episodic) >= 1

    def test_consolidation_per_user_independent(self) -> None:
        store = InMemoryStore()
        consolidator = MemoryConsolidator(
            store, turn_threshold=5, batch_size=3
        )
        for i in range(5):
            consolidator.record_turn("alice", f"alice turn {i}")
        consolidator.record_turn("bob", "bob turn")
        consolidator.record_turn("bob", "bob turn 2")

        alice_episodic = [
            m for m in store.list("alice")
            if m.metadata.get("type") == "episodic"
        ]
        bob_episodic = [
            m for m in store.list("bob")
            if m.metadata.get("type") == "episodic"
        ]
        assert len(alice_episodic) >= 1
        assert len(bob_episodic) == 0
