from __future__ import annotations

import time

from ctxeng.stores.base import ContextStore
from ctxeng.tools.base import BaseTool, ToolOutput


class FileLookupTool(BaseTool):
    name = "file_lookup"
    description = "Search stored memories by keyword. Input: a search term to find matching memories."

    def __init__(self, store: ContextStore | None = None, user_id: str = "") -> None:
        super().__init__()
        self._store = store
        self._user_id = user_id

    def set_store(self, store: ContextStore, user_id: str = "") -> None:
        self._store = store
        self._user_id = user_id

    def run(self, input_str: str) -> ToolOutput:
        start = time.perf_counter()
        if self._store is None:
            return ToolOutput(
                tool_name=self.name,
                input=input_str,
                output="Error: no memory store configured",
                success=False,
                duration_ms=(time.perf_counter() - start) * 1000,
            )
        try:
            memories = self._store.search(self._user_id, input_str)
            if not memories:
                output = "No matching memories found."
            else:
                lines = [f"- {m.text}" for m in memories[:10]]
                output = "Found memories:\n" + "\n".join(lines)
            dur = (time.perf_counter() - start) * 1000
            return ToolOutput(
                tool_name=self.name,
                input=input_str,
                output=output,
                duration_ms=dur,
            )
        except Exception as e:
            dur = (time.perf_counter() - start) * 1000
            return ToolOutput(
                tool_name=self.name,
                input=input_str,
                output=f"Error: {e}",
                success=False,
                duration_ms=dur,
            )
