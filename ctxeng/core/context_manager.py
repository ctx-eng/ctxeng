from __future__ import annotations

from typing import List, Optional

from ctxeng.assembly.assembler import ContextAssembler
from ctxeng.core.profile import ProfileStore
from ctxeng.core.safety import ContextPoisoningFilter, InputValidator, ValidationResult
from ctxeng.models import ConversationTurn
from ctxeng.stores.base import ContextStore
from ctxeng.stores.memory import InMemoryStore
from ctxeng.tools.base import ToolOutput, ToolRegistry
from ctxeng.tools.calculator import CalculatorTool
from ctxeng.tools.file_lookup import FileLookupTool
from ctxeng.tools.web_search import WebSearchTool


class ContextManager:
    def __init__(
        self,
        memory_store: Optional[ContextStore] = None,
        assembler: Optional[ContextAssembler] = None,
        max_tokens: int = 4096,
        tool_registry: Optional[ToolRegistry] = None,
        profile_store: Optional[ProfileStore] = None,
        input_validator: Optional[InputValidator] = None,
        poisoning_filter: Optional[ContextPoisoningFilter] = None,
    ) -> None:
        self.memory_store = memory_store or InMemoryStore()
        self._assembler = assembler or ContextAssembler(
            store=self.memory_store, max_tokens=max_tokens
        )
        self._tool_registry = tool_registry or self._default_tools()
        self._profile_store = profile_store or ProfileStore()
        self._input_validator = input_validator or InputValidator()
        self._poisoning_filter = poisoning_filter or ContextPoisoningFilter()

    @staticmethod
    def _default_tools() -> ToolRegistry:
        reg = ToolRegistry()
        reg.register(CalculatorTool())
        reg.register(WebSearchTool())
        return reg

    def validate_input(self, text: str) -> ValidationResult:
        return self._input_validator.validate(text)

    def filter_memories(self, memories: list) -> list:
        return self._poisoning_filter.filter_memories(memories)

    def build_prompt(
        self,
        user_id: str,
        turns: List[ConversationTurn],
        current_query: str,
        tool_outputs: Optional[List[ToolOutput]] = None,
    ) -> str:
        profile_context = self._profile_store.to_context(user_id)
        return self._assembler.assemble(
            user_id, turns, current_query,
            tool_outputs=tool_outputs,
            profile_context=profile_context,
        )

    def execute_tools(self, query: str) -> List[ToolOutput]:
        results: List[ToolOutput] = []
        matched = self._tool_registry.match(query)
        for tool in matched:
            result = tool.run(query)
            results.append(result)
        return results

    def detect_and_run_tools(self, query: str) -> List[ToolOutput]:
        return self.execute_tools(query)

    def set_file_lookup_store(self, user_id: str = "") -> None:
        ft = FileLookupTool(store=self.memory_store, user_id=user_id)
        self._tool_registry.register(ft)
