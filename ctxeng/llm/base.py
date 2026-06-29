from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LLMMessage:
    role: str
    content: str


@dataclass
class LLMResponse:
    content: str
    finish_reason: str = "stop"
    usage: dict = field(default_factory=dict)


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        ...

    @abstractmethod
    def stream(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...
