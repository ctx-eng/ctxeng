from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from ctxeng.stores.base import ContextStore


@dataclass
class LifecycleRecord:
    memory_id: str
    state: str = "born"
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 1


class LifecycleManager:
    def __init__(
        self,
        store: ContextStore,
        stale_after_turns: int = 5,
    ) -> None:
        self.store = store
        self.stale_after_turns = stale_after_turns
        self._records: dict[str, LifecycleRecord] = {}
        self._turn_counter: dict[str, int] = {}

    def record_access(self, memory_id: str) -> LifecycleRecord:
        if memory_id in self._records:
            rec = self._records[memory_id]
            rec.last_accessed = time.time()
            rec.access_count += 1
            if rec.state in ("born", "stale"):
                rec.state = "active"
        else:
            rec = LifecycleRecord(memory_id=memory_id, state="active")
            self._records[memory_id] = rec
        return rec

    def get_state(self, memory_id: str) -> Optional[str]:
        rec = self._records.get(memory_id)
        return rec.state if rec else None

    def get_record(self, memory_id: str) -> LifecycleRecord | None:
        return self._records.get(memory_id)

    def list_by_state(self, state: str) -> list[LifecycleRecord]:
        return [r for r in self._records.values() if r.state == state]

    def mark_stale(self, memory_id: str) -> None:
        if memory_id in self._records:
            self._records[memory_id].state = "stale"

    def mark_archived(self, memory_id: str) -> None:
        if memory_id in self._records:
            self._records[memory_id].state = "archived"

    def mark_dead(self, memory_id: str) -> None:
        if memory_id in self._records:
            self._records[memory_id].state = "dead"

    def archive_stale(self, user_id: str) -> int:
        archived = 0
        memories = self.store.list(user_id)
        for m in memories:
            rec = self._records.get(m.id)
            if rec and rec.state == "stale":
                m.metadata["type"] = "archived"
                self._records[m.id].state = "archived"
                archived += 1
        return archived

    def prune_dead(self) -> int:
        dead_ids = [
            mid for mid, rec in self._records.items()
            if rec.state == "dead"
        ]
        for mid in dead_ids:
            self.store.delete(mid)
            del self._records[mid]
        return len(dead_ids)
