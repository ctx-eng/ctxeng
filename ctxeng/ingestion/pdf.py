from __future__ import annotations

from pathlib import Path

from ctxeng.models import MemoryItem

try:
    from PyPDF2 import PdfReader

    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False


class PDFIngestor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest(self, filepath: str, user_id: str) -> list[MemoryItem]:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        text = self._extract_text(filepath)
        if not text.strip():
            return [
                MemoryItem(
                    user_id=user_id,
                    text=f"PDF file: {path.name} (no extractable text)",
                    metadata={"source": path.name, "type": "pdf"},
                )
            ]

        chunks = self._chunk_text(text)
        source = path.name
        return [
            MemoryItem(
                user_id=user_id,
                text=chunk,
                metadata={"source": source, "chunk": i, "type": "pdf"},
            )
            for i, chunk in enumerate(chunks)
        ]

    def _extract_text(self, filepath: str) -> str:
        if not HAS_PYPDF2:
            path = Path(filepath)
            return f"PDF file: {path.name} ({path.stat().st_size} bytes)"

        reader = PdfReader(filepath)
        pages: list[str] = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages)

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
