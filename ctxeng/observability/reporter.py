from __future__ import annotations

from typing import List

from ctxeng.observability.schema import ContextSpan, ContextTrace


def _format_trace_lines(trace: ContextTrace) -> List[str]:
    lines: List[str] = []
    lines.append(f"Context trace #{trace.trace_id}")
    lines.append(f"User: {trace.user_id}  Query: {trace.query[:60]}")
    lines.append("")

    for span in trace.spans:
        lines.append(f"  [{span.stage}]")
        details = span.summary
        if details:
            lines.append(f"    {details}")

        if span.stage == "retrieve":
            items = span.output.get("items", [])
            for item in items:
                score = item.get("score", 0)
                text = item.get("text", "")[:50]
                lines.append(f"      score={score:.3f}  \"{text}\"")

        lines.append("")

    lines.append(f"Total: {trace.total_duration_ms:.1f}ms, {trace.total_tokens} tokens, "
                  f"{len(trace.spans)} stages")
    lines.append(f"Prompt length: {len(trace.prompt)} chars")
    return lines


def format_trace(trace: ContextTrace) -> str:
    return "\n".join(_format_trace_lines(trace))


def format_trace_short(trace: ContextTrace) -> str:
    stages = ", ".join(
        f"{s.stage}: {s.duration_ms:.0f}ms" for s in trace.spans
    )
    return (f"#{trace.trace_id} | {stages} "
            f"| total {trace.total_duration_ms:.0f}ms, {trace.total_tokens}tok")
