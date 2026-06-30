from __future__ import annotations

from ctxeng.models import MemoryItem
from ctxeng.routing.diff import ContextDiff
from ctxeng.routing.lifecycle import LifecycleManager
from ctxeng.routing.router import ContextRouter
from ctxeng.stores.memory import InMemoryStore


class TestContextDiff:
    def test_empty_lists_no_diff(self) -> None:
        diff = ContextDiff.compute([], [])
        assert diff.has_changes is False
        assert diff.summary() == "No changes"

    def test_added_items_detected(self) -> None:
        prev = [MemoryItem(user_id="alice", text="old", id="m1")]
        curr = [
            MemoryItem(user_id="alice", text="old", id="m1"),
            MemoryItem(user_id="alice", text="new", id="m2"),
        ]
        diff = ContextDiff.compute(prev, curr)
        assert len(diff.added) == 1
        assert diff.added[0].id == "m2"
        assert diff.has_changes is True

    def test_removed_items_detected(self) -> None:
        prev = [
            MemoryItem(user_id="alice", text="a", id="m1"),
            MemoryItem(user_id="alice", text="b", id="m2"),
        ]
        curr = [MemoryItem(user_id="alice", text="a", id="m1")]
        diff = ContextDiff.compute(prev, curr)
        assert len(diff.removed) == 1
        assert diff.removed[0].id == "m2"

    def test_score_changes_detected(self) -> None:
        old = MemoryItem(user_id="alice", text="test", id="m1", score=0.3)
        new = MemoryItem(user_id="alice", text="test", id="m1", score=0.9)
        diff = ContextDiff.compute([old], [new])
        assert len(diff.score_changes) == 1
        assert abs(diff.score_changes["m1"] - 0.6) < 1e-6

    def test_identical_lists_no_diff(self) -> None:
        items = [MemoryItem(user_id="alice", text="a", id="m1", score=0.5)]
        diff = ContextDiff.compute(items, items)
        assert diff.has_changes is False

    def test_summary_added(self) -> None:
        diff = ContextDiff(
            added=[MemoryItem(user_id="alice", text="hello world", id="m1")],
        )
        s = diff.summary()
        assert "Added" in s
        assert "hello world" in s

    def test_summary_removed(self) -> None:
        diff = ContextDiff(
            removed=[MemoryItem(user_id="alice", text="goodbye", id="m1")],
        )
        s = diff.summary()
        assert "Removed" in s
        assert "goodbye" in s

    def test_summary_score_changes(self) -> None:
        diff = ContextDiff(score_changes={"m1": 0.5})
        s = diff.summary()
        assert "Score changes" in s


class TestLifecycleManager:
    def test_record_access_creates_born(self) -> None:
        store = InMemoryStore()
        lm = LifecycleManager(store)
        rec = lm.record_access("m1")
        assert rec.state == "active"  # first access sets to active
        assert rec.access_count == 1

    def test_record_access_updates_existing(self) -> None:
        store = InMemoryStore()
        lm = LifecycleManager(store)
        lm.record_access("m1")
        rec = lm.record_access("m1")
        assert rec.access_count == 2

    def test_get_state(self) -> None:
        store = InMemoryStore()
        lm = LifecycleManager(store)
        assert lm.get_state("nonexistent") is None
        lm.record_access("m1")
        assert lm.get_state("m1") == "active"

    def test_mark_stale(self) -> None:
        store = InMemoryStore()
        lm = LifecycleManager(store)
        lm.record_access("m1")
        lm.mark_stale("m1")
        assert lm.get_state("m1") == "stale"

    def test_mark_archived(self) -> None:
        store = InMemoryStore()
        lm = LifecycleManager(store)
        lm.record_access("m1")
        lm.mark_archived("m1")
        assert lm.get_state("m1") == "archived"

    def test_mark_dead(self) -> None:
        store = InMemoryStore()
        lm = LifecycleManager(store)
        lm.record_access("m1")
        lm.mark_dead("m1")
        assert lm.get_state("m1") == "dead"

    def test_list_by_state(self) -> None:
        store = InMemoryStore()
        lm = LifecycleManager(store)
        lm.record_access("m1")
        lm.record_access("m2")
        lm.mark_stale("m1")
        stale = lm.list_by_state("stale")
        assert len(stale) == 1
        assert stale[0].memory_id == "m1"

    def test_archive_stale(self) -> None:
        store = InMemoryStore()
        store.add("alice", "test memory")
        lm = LifecycleManager(store)
        memories = store.list("alice")
        for m in memories:
            lm.record_access(m.id)
            lm.mark_stale(m.id)
        archived = lm.archive_stale("alice")
        assert archived >= 1

    def test_prune_dead(self) -> None:
        store = InMemoryStore()
        store.add("alice", "dead memory")
        lm = LifecycleManager(store)
        memories = store.list("alice")
        lm.record_access(memories[0].id)
        lm.mark_dead(memories[0].id)
        pruned = lm.prune_dead()
        assert pruned == 1


class TestContextRouter:
    def test_step_returns_result(self) -> None:
        store = InMemoryStore()
        store.add("alice", "Alice likes concise answers.")
        router = ContextRouter(store)
        result = router.step("alice", "concise")
        assert result.step == 1
        assert len(result.memories) >= 1
        assert "concise" in result.memories[0].text

    def test_step_increments_counter(self) -> None:
        store = InMemoryStore()
        router = ContextRouter(store)
        r1 = router.step("alice", "q1")
        r2 = router.step("alice", "q2")
        r3 = router.step("alice", "q3")
        assert r1.step == 1
        assert r2.step == 2
        assert r3.step == 3

    def test_step_computes_diff_on_second_call(self) -> None:
        store = InMemoryStore()
        store.add("alice", "memory 1", metadata={"id": "m1"})
        router = ContextRouter(store)
        r1 = router.step("alice", "memory")
        # first step: previous was empty, so all memories are "added"
        assert r1.diff.has_changes is True
        assert len(r1.diff.added) == 1
        r2 = router.step("alice", "memory")
        # second step: same memories, no changes
        assert r2.diff.has_changes is False

    def test_step_unknown_user_returns_empty(self) -> None:
        store = InMemoryStore()
        store.add("alice", "test")
        router = ContextRouter(store)
        result = router.step("bob", "test")
        assert len(result.memories) == 0

    def test_reset_session_clears_state(self) -> None:
        store = InMemoryStore()
        router = ContextRouter(store)
        router.step("alice", "q1")
        assert router.get_step_count("alice") == 1
        router.reset_session("alice")
        assert router.get_step_count("alice") == 0

    def test_multiple_users_independent(self) -> None:
        store = InMemoryStore()
        router = ContextRouter(store)
        router.step("alice", "q1")
        router.step("bob", "q1")
        router.step("alice", "q2")
        assert router.get_step_count("alice") == 2
        assert router.get_step_count("bob") == 1

    def test_branch_merges_memories(self) -> None:
        store = InMemoryStore()
        store.add("alice", "base memory")
        router = ContextRouter(store)
        base = router.step("alice", "base").memories
        store.add("alice", "branch memory")
        result = router.branch("alice", "branch", base)
        assert len(result.memories) >= 2
        assert result.trace.get("branch") is True

    def test_trace_contains_metadata(self) -> None:
        store = InMemoryStore()
        store.add("alice", "test")
        router = ContextRouter(store)
        result = router.step("alice", "test")
        assert "step" in result.trace
        assert "query" in result.trace
        assert "memories_found" in result.trace

    def test_router_with_custom_retriever(self) -> None:
        store = InMemoryStore()
        store.add("alice", "custom retriever test")
        from ctxeng.retrieval.hybrid import HybridRetriever

        retriever = HybridRetriever(store)
        router = ContextRouter(store, retriever=retriever)
        result = router.step("alice", "custom")
        assert len(result.memories) >= 1

    def test_router_tracks_lifecycle(self) -> None:
        store = InMemoryStore()
        store.add("alice", "lifecycle test")
        router = ContextRouter(store)
        router.step("alice", "lifecycle")
        memories = store.list("alice")
        for m in memories:
            state = router.lifecycle.get_state(m.id)
            assert state is not None
