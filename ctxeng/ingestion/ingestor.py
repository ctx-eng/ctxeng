from __future__ import annotations

from pathlib import Path
from typing import Optional

from ctxeng.ingestion.image import ImageIngestor
from ctxeng.ingestion.structured import CSVIngestor, JSONIngestor
from ctxeng.ingestion.text import MarkdownIngestor, TextIngestor
from ctxeng.models import MemoryItem

TEXT_EXTENSIONS = {".txt", ".text", ".log", ".md", ".rst", ".py", ".js", ".ts", ".json"}


class FileIngestor:
    def __init__(self, image_caption_model: Optional[str] = None) -> None:
        self.text_ingestor = TextIngestor()
        self.markdown_ingestor = MarkdownIngestor()
        self.image_ingestor = ImageIngestor(caption_model=image_caption_model)
        self.csv_ingestor = CSVIngestor()
        self.json_ingestor = JSONIngestor()

    def ingest(self, filepath: str, user_id: str) -> list[MemoryItem]:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        suffix = path.suffix.lower()

        if suffix in {".csv"}:
            return self.csv_ingestor.ingest(filepath, user_id)
        elif suffix in {".json"}:
            return self.json_ingestor.ingest_file(filepath, user_id)
        elif suffix in {".md", ".markdown", ".rst"}:
            return self.markdown_ingestor.ingest(filepath, user_id)
        elif suffix in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}:
            return [self.image_ingestor.ingest(filepath, user_id)]
        elif suffix in TEXT_EXTENSIONS:
            return self.text_ingestor.ingest(filepath, user_id)
        else:
            return self.text_ingestor.ingest(filepath, user_id)
