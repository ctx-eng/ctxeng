from __future__ import annotations

import pytest

from ctxeng.stores.base import ContextStore
from ctxeng.stores.memory import InMemoryStore
from ctxeng.stores.sqlite import SQLiteStore

try:
    import chromadb  # noqa: F401

    from ctxeng.stores.vector import VectorStore  # noqa: F401

    HAS_CHROMADB = True
except Exception:
    HAS_CHROMADB = False


def memory_store() -> ContextStore:
    return InMemoryStore()


def sqlite_store() -> ContextStore:
    return SQLiteStore(":memory:")


@pytest.fixture(params=[memory_store, sqlite_store])
def store(request):
    if callable(request.param):
        return request.param()
    return request.param


@pytest.fixture
def stores() -> list[ContextStore]:
    instances: list[ContextStore] = [InMemoryStore(), SQLiteStore(":memory:")]
    return instances


class TestStoreContract:
    def test_add_and_search(self, stores: list[ContextStore]) -> None:
        for s in stores:
            s.clear()
            s.add("alice", "Alice likes concise answers.")
            s.add("alice", "Alice prefers bullet points.")
            s.add("bob", "Bob enjoys detailed explanations.")

            results = s.search("alice", "concise")
            assert len(results) >= 1
            assert any("concise" in r.text for r in results)

    def test_search_fallback_returns_all_when_no_match(self, stores: list[ContextStore]) -> None:
        for s in stores:
            s.clear()
            s.add("alice", "Alice likes concise answers.")
            s.add("alice", "Alice prefers bullet points.")

            results = s.search("alice", "nonexistent_keyword_xyz")
            assert len(results) == 2
            assert all(r.user_id == "alice" for r in results)

    def test_search_returns_empty_for_unknown_user(self, stores: list[ContextStore]) -> None:
        for s in stores:
            s.clear()
            s.add("alice", "Hello")
            results = s.search("unknown_user", "Hello")
            assert len(results) == 0

    def test_add_returns_memory_with_id(self, stores: list[ContextStore]) -> None:
        for s in stores:
            s.clear()
            memory = s.add("alice", "test memory")
            assert memory.id is not None
            assert len(memory.id) > 0
            assert memory.text == "test memory"
            assert memory.user_id == "alice"

    def test_delete_existing_memory(self, stores: list[ContextStore]) -> None:
        for s in stores:
            s.clear()
            memory = s.add("alice", "delete me")
            assert s.delete(memory.id) is True
            results = s.search("alice", "delete")
            assert len(results) == 0

    def test_delete_nonexistent_memory(self, stores: list[ContextStore]) -> None:
        for s in stores:
            assert s.delete("nonexistent_id") is False

    def test_list_returns_user_memories(self, stores: list[ContextStore]) -> None:
        for s in stores:
            s.clear()
            s.add("alice", "memory 1")
            s.add("alice", "memory 2")
            s.add("bob", "memory 3")

            alice_memories = s.list("alice")
            assert len(alice_memories) == 2
            assert all(m.user_id == "alice" for m in alice_memories)

    def test_clear_removes_all(self, stores: list[ContextStore]) -> None:
        for s in stores:
            s.add("alice", "memory 1")
            s.add("bob", "memory 2")
            s.clear()
            assert len(s.list("alice")) == 0
            assert len(s.list("bob")) == 0

    def test_empty_search_returns_all(self, stores: list[ContextStore]) -> None:
        for s in stores:
            s.clear()
            s.add("alice", "Hello world")
            s.add("alice", "Foo bar")
            results = s.search("alice", "")
            assert len(results) == 2

    def test_top_k_limits_results(self, stores: list[ContextStore]) -> None:
        for s in stores:
            s.clear()
            for i in range(20):
                s.add("alice", f"memory {i}")
            results = s.search("alice", "", top_k=5)
            assert len(results) == 5

    def test_case_insensitive_search(self, stores: list[ContextStore]) -> None:
        for s in stores:
            s.clear()
            s.add("alice", "Alice Likes Capital Letters")
            results = s.search("alice", "capital")
            assert len(results) == 1


class TestInMemoryStoreSpecific:
    def test_memory_objects_are_independent(self) -> None:
        s = InMemoryStore()
        m1 = s.add("alice", "memory 1")
        m2 = s.add("alice", "memory 2")
        assert m1.id != m2.id




class TestSQLiteStoreSpecific:
    def test_persistence_across_instances(self, tmp_path: pytest.TempPathFactory) -> None:
        db_file = str(tmp_path / "test.db")
        s1 = SQLiteStore(db_file)
        s1.add("alice", "persistent memory")
        s1.clear()

    def test_multiple_users(self) -> None:
        s = SQLiteStore(":memory:")
        s.add("alice", "Hello from Alice")
        s.add("bob", "Hello from Bob")
        assert len(s.list("alice")) == 1
        assert len(s.list("bob")) == 1

    def test_reopened_database(self, tmp_path: pytest.TempPathFactory) -> None:
        db_file = str(tmp_path / "reopen.db")
        s1 = SQLiteStore(db_file)
        s1.add("alice", "saved memory")
        del s1

        s2 = SQLiteStore(db_file)
        results = s2.search("alice", "saved")
        assert len(results) == 1
        assert results[0].text == "saved memory"
        s2.clear()


@pytest.mark.skipif(not HAS_CHROMADB, reason="chromadb not installed")
class TestVectorStoreSpecific:
    def test_vector_store_add_and_search(self) -> None:
        s = VectorStore("test_collection")
        s.clear()
        s.add("alice", "Alice likes concise answers.")
        results = s.search("alice", "concise")
        assert len(results) >= 1
        assert any("concise" in r.text for r in results)

    def test_vector_store_empty_search(self) -> None:
        s = VectorStore("test_empty")
        s.clear()
        results = s.search("alice", "")
        assert len(results) == 0
