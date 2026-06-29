from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


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
    def generate(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        ...

    @abstractmethod
    def stream(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...
