from __future__ import annotations

from pathlib import Path

from ctxeng.models import MemoryItem


class TextIngestor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest(self, filepath: str, user_id: str) -> list[MemoryItem]:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        content = path.read_text(encoding="utf-8")
        chunks = self._chunk_text(content)
        source = path.name

        return [
            MemoryItem(
                user_id=user_id,
                text=chunk,
                metadata={"source": source, "chunk": i, "type": "text"},
            )
            for i, chunk in enumerate(chunks)
        ]

    def _chunk_text(self, text: str) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]

        overlap = min(self.chunk_overlap, self.chunk_size // 2)
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            if end >= len(text):
                chunks.append(text[start:])
                break

            split_at = text.rfind("\n", start, end)
            if split_at <= start:
                split_at = text.rfind(" ", start, end)
            if split_at <= start:
                split_at = end

            chunks.append(text[start:split_at])
            new_start = split_at - overlap
            if new_start <= start:
                new_start = split_at
            start = new_start

        return chunks


class MarkdownIngestor(TextIngestor):
    def ingest(self, filepath: str, user_id: str) -> list[MemoryItem]:
        items = super().ingest(filepath, user_id)
        for item in items:
            item.metadata["type"] = "markdown"
        return items
