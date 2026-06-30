from __future__ import annotations

from ctxeng.assembly.prioritizer import Prioritizer
from ctxeng.assembly.templates import PromptTemplate, get_template
from ctxeng.models import ConversationTurn, MemoryItem
from ctxeng.retrieval.hybrid import HybridRetriever
from ctxeng.stores.base import ContextStore
from ctxeng.stores.memory import InMemoryStore
from ctxeng.tools.base import ToolOutput

try:
    import tiktoken

    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

_DEFAULT_ENCODING = "cl100k_base"


def _estimate_tokens(text: str) -> int:
    if HAS_TIKTOKEN:
        enc = tiktoken.get_encoding(_DEFAULT_ENCODING)
        return len(enc.encode(text))
    return max(1, len(text) // 4)


def _truncate_text(text: str, max_tokens: int) -> str:
    if _estimate_tokens(text) <= max_tokens:
        return text

    if HAS_TIKTOKEN:
        enc = tiktoken.get_encoding(_DEFAULT_ENCODING)
        tokens = enc.encode(text)
        truncated = tokens[:max_tokens]
        return enc.decode(truncated)

    # char-based fallback
    ratio = max_tokens / _estimate_tokens(text)
    cutoff = int(len(text) * ratio)
    return text[:cutoff].rsplit(" ", 1)[0] + "..."


class ContextAssembler:
    def __init__(
        self,
        store: ContextStore | None = None,
        retriever: HybridRetriever | None = None,
        template: PromptTemplate | None = None,
        max_tokens: int = 4096,
        prioritizer: Prioritizer | None = None,
    ) -> None:
        self.store = store or InMemoryStore()
        self.retriever = retriever or HybridRetriever(self.store)
        self.template = template or get_template("default")
        self.max_tokens = max_tokens
        self.prioritizer = prioritizer or Prioritizer()

    def assemble(
        self,
        user_id: str,
        turns: list[ConversationTurn],
        query: str,
        tool_outputs: list[ToolOutput] | None = None,
        profile_context: str = "",
    ) -> str:
        memories = self.retriever.search(user_id, query)
        memories = self.prioritizer.deduplicate(memories)
        memories = self.prioritizer.prioritize(memories)

        memory_block = self.prioritizer.format_memories(memories)
        history_block = self.prioritizer.format_history(turns)
        tool_block = self._format_tool_outputs(tool_outputs or [])
        profile_block = profile_context or "- no profile"

        prompt = self.template.render(
            profile=profile_block,
            memories=memory_block,
            history=history_block,
            tool_outputs=tool_block,
            query=query,
        )

        if _estimate_tokens(prompt) > self.max_tokens:
            prompt = self._trim_to_budget(
                query, prompt, memories, turns, tool_outputs or [], profile_block
            )

        return prompt

    @staticmethod
    def _format_tool_outputs(outputs: list[ToolOutput]) -> str:
        if not outputs:
            return "- none"
        lines = []
        for o in outputs:
            status = "OK" if o.success else "FAIL"
            lines.append(f"- [{status}] {o.tool_name}({o.input}): {o.output[:200]}")
        return "\n".join(lines)

    def _trim_to_budget(
        self,
        query: str,
        prompt: str,
        memories: list[MemoryItem],
        turns: list[ConversationTurn],
        tool_outputs: list[ToolOutput],
        profile_context: str = "",
    ) -> str:
        headroom = self.max_tokens - _estimate_tokens(query)
        if headroom <= 0:
            return query[: self.max_tokens * 4]

        profile_block = profile_context or "- no profile"
        prof_tokens = _estimate_tokens(profile_block)
        mem_tokens = _estimate_tokens(self.prioritizer.format_memories(memories))
        hist_tokens = _estimate_tokens(self.prioritizer.format_history(turns))
        tool_tokens = _estimate_tokens(self._format_tool_outputs(tool_outputs))
        overhead = _estimate_tokens(
            self.template.render(profile="", memories="", history="", tool_outputs="", query="")
        )

        available = headroom - overhead
        if available <= 0:
            return self.template.render(
                profile=profile_block[:200],
                memories="- none",
                history="- no prior conversation",
                tool_outputs="- none",
                query=query,
            )

        total = prof_tokens + mem_tokens + hist_tokens + tool_tokens
        if total == 0:
            return self.template.render(
                profile="- no profile",
                memories="- none",
                history="- no prior conversation",
                tool_outputs="- none",
                query=query,
            )

        prof_budget = int(available * prof_tokens / total)
        mem_budget = int(available * mem_tokens / total)
        hist_budget = int(available * hist_tokens / total)
        tool_budget = int(available * tool_tokens / total)

        profile_block = self._trim_section(profile_block, prof_budget)
        memory_block = self._trim_memories(memories, mem_budget)
        history_block = self._trim_history(turns, hist_budget)
        tool_block = self._trim_tool_outputs(tool_outputs, tool_budget)

        return self.template.render(
            profile=profile_block,
            memories=memory_block,
            history=history_block,
            tool_outputs=tool_block,
            query=query,
        )

    @staticmethod
    def _trim_section(text: str, budget: int) -> str:
        if budget <= 0 or not text:
            return "- no profile"
        if _estimate_tokens(text) <= budget:
            return text
        lines = text.split("\n")
        kept: list[str] = []
        for line in lines:
            if _estimate_tokens(line) > budget:
                break
            kept.append(line)
            budget -= _estimate_tokens(line)
        return "\n".join(kept) if kept else "- no profile"

    @staticmethod
    def _trim_tool_outputs(outputs: list[ToolOutput], budget: int) -> str:
        if budget <= 0 or not outputs:
            return "- none"
        lines = []
        for o in outputs:
            status = "OK" if o.success else "FAIL"
            line = f"- [{status}] {o.tool_name}({o.input}): {o.output[:200]}"
            if _estimate_tokens(line) > budget:
                break
            lines.append(line)
            budget -= _estimate_tokens(line)
        if not lines:
            return "- none"
        return "\n".join(lines)

    def _trim_memories(self, memories: list[MemoryItem], budget: int) -> str:
        if budget <= 0:
            return "- none"

        lines = []
        for m in memories:
            line = f"- {m.text}"
            if _estimate_tokens(line) > budget:
                break
            lines.append(line)
            budget -= _estimate_tokens(line)

        if not lines:
            return "- none"
        return "\n".join(lines)

    def _trim_history(self, turns: list[ConversationTurn], budget: int) -> str:
        if budget <= 0:
            return "- no prior conversation"

        lines = []
        for t in reversed(turns):
            line = f"{t.role}: {t.content}"
            if _estimate_tokens(line) > budget:
                break
            lines.append(line)
            budget -= _estimate_tokens(line)

        if not lines:
            return "- no prior conversation"

        lines.reverse()
        return "\n".join(lines)
