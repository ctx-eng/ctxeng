from __future__ import annotations

import time

from ctxeng.assembly.assembler import ContextAssembler, _estimate_tokens
from ctxeng.models import ConversationTurn
from ctxeng.observability.schema import ContextSpan, ContextTrace
from ctxeng.tools.base import ToolOutput

_trace_store: dict[str, ContextTrace] = {}


def get_trace(trace_id: str) -> ContextTrace | None:
    return _trace_store.get(trace_id)


def list_traces(user_id: str | None = None, limit: int = 10) -> list[ContextTrace]:
    traces = list(_trace_store.values())
    if user_id:
        traces = [t for t in traces if t.user_id == user_id]
    traces.sort(key=lambda t: t.total_duration_ms, reverse=True)
    return traces[:limit]


class ContextTracer:
    def __init__(self, assembler: ContextAssembler) -> None:
        self.assembler = assembler

    def assemble(
        self,
        user_id: str,
        turns: list[ConversationTurn],
        query: str,
        tool_outputs: list[ToolOutput] | None = None,
        profile_context: str = "",
    ) -> tuple[str, ContextTrace]:
        trace = ContextTrace(
            user_id=user_id,
            query=query,
        )

        # Span 1: Retrieve
        t0 = time.perf_counter()
        memories = self.assembler.retriever.search(user_id, query)
        retrieve_ms = (time.perf_counter() - t0) * 1000
        retrieve_tokens = sum(_estimate_tokens(m.text) for m in memories)
        trace.add_span(
            ContextSpan(
                stage="retrieve",
                input={"user_id": user_id, "query": query, "top_k": 10},
                output={
                    "item_count": len(memories),
                    "items": [{"text": m.text, "score": m.score} for m in memories],
                },
                duration_ms=retrieve_ms,
                token_count=retrieve_tokens,
            )
        )

        # Span 2: Deduplicate
        t0 = time.perf_counter()
        deduped = self.assembler.prioritizer.deduplicate(memories)
        dedup_ms = (time.perf_counter() - t0) * 1000
        trace.add_span(
            ContextSpan(
                stage="deduplicate",
                input={"item_count": len(memories)},
                output={"item_count": len(deduped), "removed": len(memories) - len(deduped)},
                duration_ms=dedup_ms,
            )
        )

        # Span 3: Prioritize
        t0 = time.perf_counter()
        prioritized = self.assembler.prioritizer.prioritize(deduped)
        prioritize_ms = (time.perf_counter() - t0) * 1000
        trace.add_span(
            ContextSpan(
                stage="prioritize",
                input={"item_count": len(deduped)},
                output={"item_count": len(prioritized)},
                duration_ms=prioritize_ms,
            )
        )

        # Span 4: Tool execution
        tool_outputs = tool_outputs or []
        tool_ms = 0.0
        if tool_outputs:
            t0 = time.perf_counter()
            tool_ms = (time.perf_counter() - t0) * 1000
            trace.add_span(
                ContextSpan(
                    stage="tool_execution",
                    input={"tool_count": len(tool_outputs)},
                    output={
                        "tools": [
                            {
                                "name": t.tool_name,
                                "success": t.success,
                                "duration_ms": round(t.duration_ms, 1),
                            }
                            for t in tool_outputs
                        ]
                    },
                    duration_ms=tool_ms,
                )
            )

        # Span 5: Render & check budget
        profile_block = profile_context or "- no profile"
        memory_block = self.assembler.prioritizer.format_memories(prioritized)
        history_block = self.assembler.prioritizer.format_history(turns)
        tool_block = self.assembler._format_tool_outputs(tool_outputs)

        t0 = time.perf_counter()
        prompt = self.assembler.template.render(
            profile=profile_block,
            memories=memory_block,
            history=history_block,
            tool_outputs=tool_block,
            query=query,
        )
        prompt_tokens = _estimate_tokens(prompt)
        over_budget = prompt_tokens > self.assembler.max_tokens
        render_ms = (time.perf_counter() - t0) * 1000

        # Span 6: Trim if over budget
        trim_ms = 0.0
        if over_budget:
            t0 = time.perf_counter()
            prompt = self.assembler._trim_to_budget(
                query, prompt, prioritized, turns, tool_outputs, profile_block
            )
            prompt_tokens = _estimate_tokens(prompt)
            trim_ms = (time.perf_counter() - t0) * 1000

        trace.add_span(
            ContextSpan(
                stage="assemble",
                input={
                    "max_tokens": self.assembler.max_tokens,
                    "memory_count": len(prioritized),
                    "history_turns": len(turns),
                    "tool_count": len(tool_outputs),
                },
                output={
                    "prompt_tokens": prompt_tokens,
                    "over_budget": over_budget,
                    "trim_ms": round(trim_ms, 1),
                },
                duration_ms=render_ms + trim_ms,
                token_count=prompt_tokens,
            )
        )

        trace.prompt = prompt
        trace.total_tokens = prompt_tokens
        _trace_store[trace.trace_id] = trace

        return prompt, trace
