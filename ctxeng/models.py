from dataclasses import dataclass
from typing import Optional


@dataclass
class ConversationTurn:
    role: str
    content: str


@dataclass
class MemoryItem:
    text: str
    user_id: str
    timestamp: Optional[str] = None
