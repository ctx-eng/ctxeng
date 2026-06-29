from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from ctxeng.ingestion.image import ImageIngestor
from ctxeng.ingestion.ingestor import FileIngestor
from ctxeng.ingestion.structured import CSVIngestor, JSONIngestor
from ctxeng.ingestion.text import MarkdownIngestor, TextIngestor

try:
    from PIL import Image as PILImage

    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False

try:
    import transformers  # noqa: F401

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class TestTextIngestor:
    def test_ingest_txt_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("Hello world.\nThis is a test.\n")
        ingestor = TextIngestor()
        items = ingestor.ingest(str(f), "alice")
        assert len(items) >= 1
        assert items[0].text == "Hello world.\nThis is a test.\n"
        assert items[0].user_id == "alice"
        assert items[0].metadata["source"] == "test.txt"
        assert items[0].metadata["type"] == "text"

    def test_ingest_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"
        f.write_text("")
        ingestor = TextIngestor()
        items = ingestor.ingest(str(f), "alice")
        assert len(items) == 1
        assert items[0].text == ""

    def test_chunking_large_file(self, tmp_path: Path) -> None:
        f = tmp_path / "large.txt"
        f.write_text("word " * 5000)
        ingestor = TextIngestor(chunk_size=100)
        items = ingestor.ingest(str(f), "alice")
        assert len(items) > 1
        for item in items:
            assert len(item.text) <= 150  # chunk_size + some overlap

    def test_file_not_found(self) -> None:
        ingestor = TextIngestor()
        with pytest.raises(FileNotFoundError):
            ingestor.ingest("/nonexistent/file.txt", "alice")


class TestMarkdownIngestor:
    def test_ingest_markdown(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.md"
        f.write_text("# Title\n\nSome content here.\n")
        ingestor = MarkdownIngestor()
        items = ingestor.ingest(str(f), "alice")
        assert len(items) >= 1
        assert items[0].metadata["type"] == "markdown"
        assert items[0].metadata["source"] == "doc.md"


class TestCSVIngestor:
    def test_ingest_csv(self, tmp_path: Path) -> None:
        f = tmp_path / "data.csv"
        f.write_text("name,age,city\nAlice,30,NYC\nBob,25,LA\n")
        ingestor = CSVIngestor()
        items = ingestor.ingest(str(f), "alice")
        assert len(items) >= 1
        summary = items[0]
        assert "CSV file" in summary.text
        assert "2 rows" in summary.text

    def test_ingest_empty_csv(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.csv"
        f.write_text("name,age\n")
        ingestor = CSVIngestor()
        items = ingestor.ingest(str(f), "alice")
        assert len(items) == 0

    def test_ingest_csv_max_rows(self, tmp_path: Path) -> None:
        f = tmp_path / "many.csv"
        lines = ["col"] + [f"val{i}" for i in range(100)]
        f.write_text("\n".join(lines))
        ingestor = CSVIngestor()
        items = ingestor.ingest(str(f), "alice", max_rows=5)
        assert len(items) <= 7  # 1 summary + up to 5 rows


class TestJSONIngestor:
    def test_ingest_dict(self) -> None:
        ingestor = JSONIngestor()
        data = {"name": "Alice", "age": 30, "city": "NYC"}
        items = ingestor.ingest(data, "alice", source_name="user_profile")
        assert len(items) == 3
        assert any("name: Alice" in i.text for i in items)
        assert all(i.metadata["type"] == "json_field" for i in items)

    def test_ingest_list(self) -> None:
        ingestor = JSONIngestor()
        data = [{"id": 1, "title": "First"}, {"id": 2, "title": "Second"}]
        items = ingestor.ingest(data, "alice")
        assert len(items) >= 1
        assert any("JSON array" in i.text for i in items)

    def test_ingest_file(self, tmp_path: Path) -> None:
        f = tmp_path / "data.json"
        json.dump({"key": "value"}, f.open("w"))
        ingestor = JSONIngestor()
        items = ingestor.ingest_file(str(f), "alice")
        assert len(items) >= 1

    def test_ingest_nested_dict(self) -> None:
        ingestor = JSONIngestor()
        data = {"user": {"name": "Alice", "settings": {"theme": "dark"}}}
        items = ingestor.ingest(data, "alice")
        assert any("object with" in i.text for i in items)

    def test_ingest_null_values(self) -> None:
        ingestor = JSONIngestor()
        items = ingestor.ingest({"key": None}, "alice")
        assert any("null" in i.text for i in items)

    def test_ingest_bool_values(self) -> None:
        ingestor = JSONIngestor()
        items = ingestor.ingest({"active": True}, "alice")
        assert any("True" in i.text for i in items)


class TestImageIngestor:
    def test_basic_info_without_pillow(self) -> None:
        ingestor = ImageIngestor()
        assert ingestor.can_caption is False

    @pytest.mark.skipif(not HAS_PILLOW, reason="Pillow not installed")
    def test_ingest_image(self, tmp_path: Path) -> None:
        img_path = tmp_path / "test.png"
        img = PILImage.new("RGB", (100, 50), color="red")
        img.save(str(img_path))

        ingestor = ImageIngestor()
        memory = ingestor.ingest(str(img_path), "alice")
        assert memory.user_id == "alice"
        assert "test.png" in memory.text
        assert memory.metadata["type"] == "image"

    @pytest.mark.skipif(not HAS_PILLOW, reason="Pillow not installed")
    def test_image_format_and_size_in_text(self, tmp_path: Path) -> None:
        img_path = tmp_path / "photo.jpg"
        img = PILImage.new("RGB", (200, 100), color="blue")
        img.save(str(img_path), "JPEG")

        ingestor = ImageIngestor()
        memory = ingestor.ingest(str(img_path), "alice")
        assert "200x100" in memory.text
        assert "JPEG" in memory.text or "jpeg" in memory.text.lower()

    @pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
    def test_image_captioning(self, tmp_path: Path) -> None:
        img_path = tmp_path / "test.png"
        img = PILImage.new("RGB", (50, 50), color="green")
        img.save(str(img_path))

        ingestor = ImageIngestor(caption_model="dummy")
        memory = ingestor.ingest(str(img_path), "alice")
        assert memory.user_id == "alice"


class TestFileIngestor:
    def test_text_file_dispatched(self, tmp_path: Path) -> None:
        f = tmp_path / "notes.txt"
        f.write_text("hello")
        ingestor = FileIngestor()
        items = ingestor.ingest(str(f), "alice")
        assert len(items) == 1

    def test_markdown_file_dispatched(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.md"
        f.write_text("# title")
        ingestor = FileIngestor()
        items = ingestor.ingest(str(f), "alice")
        assert items[0].metadata["type"] == "markdown"

    def test_csv_file_dispatched(self, tmp_path: Path) -> None:
        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2\n")
        ingestor = FileIngestor()
        items = ingestor.ingest(str(f), "alice")
        assert any("CSV file" in i.text for i in items)

    def test_json_file_dispatched(self, tmp_path: Path) -> None:
        f = tmp_path / "data.json"
        json.dump({"key": "val"}, f.open("w"))
        ingestor = FileIngestor()
        items = ingestor.ingest(str(f), "alice")
        assert len(items) >= 1

    def test_image_file_dispatched(self, tmp_path: Path) -> None:
        if not HAS_PILLOW:
            pytest.skip("Pillow not installed")
        img_path = tmp_path / "photo.png"
        img = PILImage.new("RGB", (10, 10))
        img.save(str(img_path))
        ingestor = FileIngestor()
        items = ingestor.ingest(str(img_path), "alice")
        assert len(items) == 1
        assert items[0].metadata["type"] == "image"

    def test_unknown_extension(self, tmp_path: Path) -> None:
        f = tmp_path / "data.xyz"
        f.write_text("unknown format")
        ingestor = FileIngestor()
        items = ingestor.ingest(str(f), "alice")
        assert len(items) >= 1

    def test_file_not_found(self) -> None:
        ingestor = FileIngestor()
        with pytest.raises(FileNotFoundError):
            ingestor.ingest("/nonexistent/file.xyz", "alice")
