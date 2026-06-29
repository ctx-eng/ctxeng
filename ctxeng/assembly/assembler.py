from __future__ import annotations

from typing import List, Optional

from ctxeng.assembly.prioritizer import Prioritizer
from ctxeng.assembly.templates import PromptTemplate, get_template
from ctxeng.models import ConversationTurn, MemoryItem
from ctxeng.retrieval.hybrid import HybridRetriever
from ctxeng.stores.base import ContextStore
from ctxeng.stores.memory import InMemoryStore

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
        store: Optional[ContextStore] = None,
        retriever: Optional[HybridRetriever] = None,
        template: Optional[PromptTemplate] = None,
        max_tokens: int = 4096,
        prioritizer: Optional[Prioritizer] = None,
    ) -> None:
        self.store = store or InMemoryStore()
        self.retriever = retriever or HybridRetriever(self.store)
        self.template = template or get_template("default")
        self.max_tokens = max_tokens
        self.prioritizer = prioritizer or Prioritizer()

    def assemble(
        self,
        user_id: str,
        turns: List[ConversationTurn],
        query: str,
    ) -> str:
        memories = self.retriever.search(user_id, query)
        memories = self.prioritizer.deduplicate(memories)
        memories = self.prioritizer.prioritize(memories)

        memory_block = self.prioritizer.format_memories(memories)
        history_block = self.prioritizer.format_history(turns)

        prompt = self.template.render(
            memories=memory_block,
            history=history_block,
            query=query,
        )

        if _estimate_tokens(prompt) > self.max_tokens:
            prompt = self._trim_to_budget(query, prompt, memories, turns)

        return prompt

    def _trim_to_budget(
        self,
        query: str,
        prompt: str,
        memories: List[MemoryItem],
        turns: List[ConversationTurn],
    ) -> str:
        headroom = self.max_tokens - _estimate_tokens(query)
        if headroom <= 0:
            return query[: self.max_tokens * 4]

        mem_tokens = _estimate_tokens(self.prioritizer.format_memories(memories))
        hist_tokens = _estimate_tokens(self.prioritizer.format_history(turns))
        overhead = _estimate_tokens(
            self.template.render(memories="", history="", query="")
        )

        available = headroom - overhead
        if available <= 0:
            return self.template.render(
                memories="- none",
                history="- no prior conversation",
                query=query,
            )

        # allocate proportionally
        total = mem_tokens + hist_tokens
        if total == 0:
            return self.template.render(
                memories="- none",
                history="- no prior conversation",
                query=query,
            )

        mem_budget = int(available * mem_tokens / total)
        hist_budget = int(available * hist_tokens / total)

        memory_block = self._trim_memories(memories, mem_budget)
        history_block = self._trim_history(turns, hist_budget)

        return self.template.render(
            memories=memory_block,
            history=history_block,
            query=query,
        )

    def _trim_memories(self, memories: List[MemoryItem], budget: int) -> str:
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

    def _trim_history(self, turns: List[ConversationTurn], budget: int) -> str:
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
