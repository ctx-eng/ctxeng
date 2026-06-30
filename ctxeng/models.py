from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4


@dataclass
class ConversationTurn:
    role: str
    content: str


@dataclass
class MemoryItem:
    user_id: str
    text: str
    timestamp: str | None = None
    id: str = field(default_factory=lambda: uuid4().hex)
    metadata: dict = field(default_factory=dict)
    score: float = 0.0
