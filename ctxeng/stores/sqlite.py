from __future__ import annotations

import json
import sqlite3
import threading
from typing import Optional

from ctxeng.models import MemoryItem
from ctxeng.stores.base import ContextStore


class SQLiteStore(ContextStore):
    def __init__(self, db_path: str = ":memory:") -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                text TEXT NOT NULL,
                timestamp TEXT,
                metadata TEXT DEFAULT '{}'
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_user_id ON memories (user_id)"
        )
        self._conn.commit()

    def add(self, user_id: str, text: str, metadata: Optional[dict] = None) -> MemoryItem:
        memory = MemoryItem(
            user_id=user_id,
            text=text,
            metadata=metadata or {},
        )
        with self._lock:
            self._conn.execute(
                "INSERT INTO memories (id, user_id, text, timestamp, metadata) VALUES (?, ?, ?, ?, ?)",
                (memory.id, memory.user_id, memory.text, memory.timestamp, json.dumps(memory.metadata)),
            )
            self._conn.commit()
        return memory

    def search(self, user_id: str, query: str, top_k: int = 10) -> list[MemoryItem]:
        with self._lock:
            if query:
                rows = self._conn.execute(
                    "SELECT id, user_id, text, timestamp, metadata FROM memories "
                    "WHERE user_id = ? AND LOWER(text) LIKE ? LIMIT ?",
                    (user_id, f"%{query.lower()}%", top_k),
                ).fetchall()
            if not query or not rows:
                rows = self._conn.execute(
                    "SELECT id, user_id, text, timestamp, metadata FROM memories "
                    "WHERE user_id = ? LIMIT ?",
                    (user_id, top_k),
                ).fetchall()
        results = []
        for row in rows:
            memory = MemoryItem(
                id=row[0],
                user_id=row[1],
                text=row[2],
                timestamp=row[3],
                metadata=json.loads(row[4]) if row[4] else {},
                score=0.5,
            )
            results.append(memory)
        return results

    def delete(self, memory_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            self._conn.commit()
            return cursor.rowcount > 0

    def list(self, user_id: str) -> list[MemoryItem]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, user_id, text, timestamp, metadata FROM memories WHERE user_id = ?",
                (user_id,),
            ).fetchall()
        return [
            MemoryItem(
                id=row[0],
                user_id=row[1],
                text=row[2],
                timestamp=row[3],
                metadata=json.loads(row[4]) if row[4] else {},
            )
            for row in rows
        ]

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM memories")
            self._conn.commit()
