from __future__ import annotations

from ctxeng.models import MemoryItem
from ctxeng.stores.base import ContextStore


class VectorStore(ContextStore):
    def __init__(
        self, collection_name: str = "ctxeng", persist_directory: str | None = None
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError(
                "chromadb is required for VectorStore. Install with: pip install chromadb"
            ) from exc
        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.EphemeralClient()
        self._collection = self._client.get_or_create_collection(collection_name)

    def add(self, user_id: str, text: str, metadata: dict | None = None) -> MemoryItem:
        memory = MemoryItem(
            user_id=user_id,
            text=text,
            metadata=metadata or {},
        )
        self._collection.add(
            ids=[memory.id],
            documents=[memory.text],
            metadatas=[{"user_id": memory.user_id, **(memory.metadata)}],
        )
        return memory

    def search(self, user_id: str, query: str, top_k: int = 10) -> list[MemoryItem]:
        results = self._collection.query(
            query_texts=[query],
            n_results=top_k,
            where={"user_id": user_id},
        )
        items = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                doc = results["documents"][0][i] if results["documents"] else ""
                dist = results["distances"][0][i] if results["distances"] else 0.0
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                # chromadb returns distance (lower = closer); convert to similarity score
                score = max(0.0, 1.0 - dist / 2.0)
                items.append(
                    MemoryItem(
                        id=doc_id,
                        user_id=user_id,
                        text=doc,
                        metadata=meta,
                        score=score,
                    )
                )
        return items

    def delete(self, memory_id: str) -> bool:
        try:
            self._collection.delete(ids=[memory_id])
            return True
        except Exception:
            return False

    def list(self, user_id: str) -> list[MemoryItem]:
        results = self._collection.get(where={"user_id": user_id})
        items = []
        if results["ids"]:
            for i, doc_id in enumerate(results["ids"]):
                doc = results["documents"][i] if results["documents"] else ""
                meta = results["metadatas"][i] if results["metadatas"] else {}
                items.append(
                    MemoryItem(
                        id=doc_id,
                        user_id=user_id,
                        text=doc,
                        metadata=meta,
                    )
                )
        return items

    def clear(self) -> None:
        self._collection.delete(where={})
