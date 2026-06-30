from __future__ import annotations

from pathlib import Path

from ctxeng.ingestion.image import ImageIngestor
from ctxeng.ingestion.pdf import PDFIngestor
from ctxeng.ingestion.structured import CSVIngestor, JSONIngestor
from ctxeng.ingestion.text import MarkdownIngestor, TextIngestor
from ctxeng.models import MemoryItem

try:
    import magic as libmagic  # type: ignore

    HAS_PYTHON_MAGIC = True
except ImportError:
    HAS_PYTHON_MAGIC = False

TEXT_EXTENSIONS = {".txt", ".text", ".log", ".md", ".rst", ".py", ".js", ".ts", ".json"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
PDF_EXTENSIONS = {".pdf"}

_MIME_TEXT_PREFIXES = ("text/", "application/json", "application/xml")
_MIME_IMAGE_PREFIXES = ("image/",)
_MIME_PDF = "application/pdf"


def _detect_mime(filepath: str) -> str | None:
    if not HAS_PYTHON_MAGIC:
        return None
    try:
        return libmagic.from_file(filepath, mime=True)
    except Exception:
        return None


class FileIngestor:
    def __init__(self, image_caption_model: str | None = None) -> None:
        self.text_ingestor = TextIngestor()
        self.markdown_ingestor = MarkdownIngestor()
        self.image_ingestor = ImageIngestor(caption_model=image_caption_model)
        self.csv_ingestor = CSVIngestor()
        self.json_ingestor = JSONIngestor()
        self.pdf_ingestor = PDFIngestor()

    def ingest(self, filepath: str, user_id: str) -> list[MemoryItem]:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        mime = _detect_mime(filepath)
        if mime and mime.startswith(_MIME_IMAGE_PREFIXES):
            return [self.image_ingestor.ingest(filepath, user_id)]
        if mime and mime == _MIME_PDF:
            return self.pdf_ingestor.ingest(filepath, user_id)
        if mime and mime.startswith(_MIME_TEXT_PREFIXES):
            suffix = path.suffix.lower()
            if suffix in {".csv"}:
                return self.csv_ingestor.ingest(filepath, user_id)
            return self.text_ingestor.ingest(filepath, user_id)

        suffix = path.suffix.lower()
        if suffix in {".csv"}:
            return self.csv_ingestor.ingest(filepath, user_id)
        elif suffix in {".json"}:
            return self.json_ingestor.ingest_file(filepath, user_id)
        elif suffix in {".md", ".markdown", ".rst"}:
            return self.markdown_ingestor.ingest(filepath, user_id)
        elif suffix in IMAGE_EXTENSIONS:
            return [self.image_ingestor.ingest(filepath, user_id)]
        elif suffix in PDF_EXTENSIONS:
            return self.pdf_ingestor.ingest(filepath, user_id)
        elif suffix in TEXT_EXTENSIONS:
            return self.text_ingestor.ingest(filepath, user_id)
        else:
            return self.text_ingestor.ingest(filepath, user_id)
