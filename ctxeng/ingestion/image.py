from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from ctxeng.models import MemoryItem

try:
    from PIL import Image as PILImage

    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False


class ImageIngestor:
    def __init__(self, caption_model: Optional[str] = None) -> None:
        self.caption_model = caption_model
        self._hf_pipeline = None

    @property
    def can_caption(self) -> bool:
        if self.caption_model:
            try:
                import transformers  # noqa: F401
                return True
            except ImportError:
                return False
        return False

    def _load_captioner(self) -> None:
        if self._hf_pipeline is not None:
            return
        from transformers import pipeline as hf_pipeline

        model = self.caption_model or "Salesforce/blip-image-captioning-base"
        self._hf_pipeline = hf_pipeline("image-to-text", model=model)

    def _extract_basic_info(self, filepath: str) -> str:
        if not HAS_PILLOW:
            return f"Image file: {Path(filepath).name}"
        img = PILImage.open(filepath)
        parts = [
            f"Image: {Path(filepath).name}",
            f"Format: {img.format}",
            f"Size: {img.size[0]}x{img.size[1]}",
            f"Mode: {img.mode}",
        ]
        img.close()
        return " | ".join(parts)

    def _generate_caption(self, filepath: str) -> str:
        if not self.can_caption:
            return self._extract_basic_info(filepath)
        self._load_captioner()
        from PIL import Image as PILImage
        img = PILImage.open(filepath)
        result = self._hf_pipeline(img)
        img.close()
        caption = result[0]["generated_text"] if result else ""
        return f"{self._extract_basic_info(filepath)} | Caption: {caption}"

    def ingest(self, filepath: str, user_id: str) -> MemoryItem:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        text = self._generate_caption(filepath)
        return MemoryItem(
            user_id=user_id,
            text=text,
            metadata={"source": path.name, "type": "image"},
        )
