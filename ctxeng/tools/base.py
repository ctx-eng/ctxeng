from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolOutput:
    tool_name: str
    input: str
    output: str
    success: bool = True
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseTool(ABC):
    name: str = ""
    description: str = ""

    @abstractmethod
    def run(self, input_str: str) -> ToolOutput:
        ...

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": "Tool input"
                        }
                    },
                    "required": ["input"]
                }
            }
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def list(self) -> list[str]:
        return list(self._tools.keys())

    def execute(self, name: str, input_str: str) -> ToolOutput | None:
        tool = self.get(name)
        if tool is None:
            return None
        return tool.run(input_str)

    def match(self, query: str) -> list[BaseTool]:
        """Return tools whose name or keywords appear in the query."""
        matched: list[BaseTool] = []
        q = query.lower()
        for tool in self._tools.values():
            if tool.name.lower() in q:
                matched.append(tool)
        return matched
