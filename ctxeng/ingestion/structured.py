from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ctxeng.models import MemoryItem


class CSVIngestor:
    def ingest(self, filepath: str, user_id: str, max_rows: int = 20) -> List[MemoryItem]:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            return []

        items: List[MemoryItem] = []
        source = path.name

        summary = f"CSV file: {source} | {len(rows)} rows | Columns: {', '.join(reader.fieldnames or [])}"
        items.append(MemoryItem(
            user_id=user_id,
            text=summary,
            metadata={"source": source, "type": "csv_summary"},
        ))

        for i, row in enumerate(rows[:max_rows]):
            desc = "; ".join(f"{k}: {v}" for k, v in row.items() if v)
            items.append(MemoryItem(
                user_id=user_id,
                text=desc,
                metadata={"source": source, "type": "csv_row", "row": i},
            ))

        return items


class JSONIngestor:
    def ingest(
        self,
        data: Any,
        user_id: str,
        source_name: str = "json_data",
        max_items: int = 20,
    ) -> List[MemoryItem]:
        items: List[MemoryItem] = []

        if isinstance(data, dict):
            for key, value in data.items():
                desc = self._describe(key, value)
                items.append(MemoryItem(
                    user_id=user_id,
                    text=desc,
                    metadata={"source": source_name, "type": "json_field", "field": key},
                ))
                if len(items) >= max_items:
                    break

        elif isinstance(data, list):
            summary = f"JSON array: {len(data)} items from {source_name}"
            items.append(MemoryItem(
                user_id=user_id,
                text=summary,
                metadata={"source": source_name, "type": "json_summary"},
            ))
            for i, item in enumerate(data[:max_items]):
                desc = self._describe(f"item_{i}", item)
                items.append(MemoryItem(
                    user_id=user_id,
                    text=desc,
                    metadata={"source": source_name, "type": "json_item", "index": i},
                ))

        return items

    def ingest_file(self, filepath: str, user_id: str) -> List[MemoryItem]:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return self.ingest(data, user_id, source_name=path.name)

    def _describe(self, key: str, value: Any, max_length: int = 200) -> str:
        if isinstance(value, dict):
            return f"{key}: object with {len(value)} fields"
        elif isinstance(value, list):
            return f"{key}: array of {len(value)} items"
        elif isinstance(value, bool):
            return f"{key}: {value}"
        elif value is None:
            return f"{key}: null"
        else:
            text = str(value)[:max_length]
            return f"{key}: {text}"
