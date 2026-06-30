from __future__ import annotations

from ctxeng.models import MemoryItem
from ctxeng.stores.base import ContextStore


class TieredStore(ContextStore):
    """Composable store that queries tiers in priority order (working -> episodic -> long-term).

    Falls through tiers until top_k results are found.
    """

    def __init__(
        self,
        working_store: ContextStore,
        episodic_store: ContextStore,
        long_term_store: ContextStore,
    ) -> None:
        self.working = working_store
        self.episodic = episodic_store
        self.long_term = long_term_store

    def add(self, user_id: str, text: str, metadata: dict | None = None) -> MemoryItem:
        tier = (metadata or {}).get("type", "working")
        if tier == "long_term":
            return self.long_term.add(user_id, text, metadata)
        elif tier == "episodic":
            return self.episodic.add(user_id, text, metadata)
        return self.working.add(user_id, text, metadata)

    def search(self, user_id: str, query: str, top_k: int = 10) -> list[MemoryItem]:
        results: list[MemoryItem] = []
        seen_ids: set[str] = set()

        for store in (self.working, self.episodic, self.long_term):
            if len(results) >= top_k:
                break
            needed = top_k - len(results)
            batch = store.search(user_id, query, top_k=needed)
            for item in batch:
                if item.id not in seen_ids:
                    seen_ids.add(item.id)
                    results.append(item)

        return results[:top_k]

    def delete(self, memory_id: str) -> bool:
        for store in (self.working, self.episodic, self.long_term):
            if store.delete(memory_id):
                return True
        return False

    def list(self, user_id: str) -> list[MemoryItem]:
        seen_ids: set[str] = set()
        all_items: list[MemoryItem] = []
        for store in (self.working, self.episodic, self.long_term):
            for item in store.list(user_id):
                if item.id not in seen_ids:
                    seen_ids.add(item.id)
                    all_items.append(item)
        return all_items

    def clear(self) -> None:
        self.working.clear()
        self.episodic.clear()
        self.long_term.clear()
