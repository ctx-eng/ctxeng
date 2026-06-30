from __future__ import annotations

import time
from urllib.parse import urlparse

from ctxeng.tools.base import BaseTool, ToolOutput


class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Fetch content from a URL or perform a basic web lookup. Input: a URL starting with http:// or https://"

    def run(self, input_str: str) -> ToolOutput:
        start = time.perf_counter()
        url = input_str.strip()
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            dur = (time.perf_counter() - start) * 1000
            return ToolOutput(
                tool_name=self.name,
                input=input_str,
                output="Error: provide a valid URL starting with http:// or https://",
                success=False,
                duration_ms=dur,
            )
        try:
            import urllib.request

            req = urllib.request.Request(url, headers={"User-Agent": "CtxEng/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                content = resp.read().decode("utf-8", errors="replace")[:3000]
            dur = (time.perf_counter() - start) * 1000
            return ToolOutput(
                tool_name=self.name,
                input=input_str,
                output=content,
                duration_ms=dur,
            )
        except Exception as e:
            dur = (time.perf_counter() - start) * 1000
            return ToolOutput(
                tool_name=self.name,
                input=input_str,
                output=f"Error fetching URL: {e}",
                success=False,
                duration_ms=dur,
            )
