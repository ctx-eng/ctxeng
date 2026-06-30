from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4


@dataclass
class ContextSpan:
    stage: str
    input: dict
    output: dict
    duration_ms: float
    token_count: int = 0
    parent_id: str | None = None
    id: str = field(default_factory=lambda: uuid4().hex)

    @property
    def summary(self) -> str:
        items = self.output.get("item_count", 0)
        tokens = self.token_count
        ms = self.duration_ms
        details = ""
        if items:
            details += f"{items} items, "
        if tokens:
            details += f"{tokens} tokens, "
        return f"{self.stage}: {details}{ms:.1f}ms"


@dataclass
class ContextTrace:
    user_id: str
    query: str
    spans: list[ContextSpan] = field(default_factory=list)
    trace_id: str = field(default_factory=lambda: uuid4().hex)
    total_duration_ms: float = 0.0
    total_tokens: int = 0
    prompt: str = ""

    def add_span(self, span: ContextSpan) -> None:
        self.spans.append(span)
        self.total_duration_ms += span.duration_ms
        self.total_tokens += span.token_count
