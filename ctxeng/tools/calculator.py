from __future__ import annotations

import time

from ctxeng.tools.base import BaseTool, ToolOutput


class CalculatorTool(BaseTool):
    name = "calculator"
    description = (
        "Evaluate a mathematical expression. Input: a math expression like '2 + 3 * 4'."
        " Supports +, -, *, /, **, %, //, and parentheses."
    )

    def run(self, input_str: str) -> ToolOutput:
        start = time.perf_counter()
        allowed = set("0123456789.+-*/()% ")
        filtered = "".join(c for c in input_str if c in allowed).strip()
        if not filtered:
            dur = (time.perf_counter() - start) * 1000
            return ToolOutput(
                tool_name=self.name,
                input=input_str,
                output="Error: no mathematical expression found",
                success=False,
                duration_ms=dur,
            )
        try:
            result = eval(filtered, {"__builtins__": {}}, {})
            dur = (time.perf_counter() - start) * 1000
            return ToolOutput(
                tool_name=self.name,
                input=input_str,
                output=str(result),
                duration_ms=dur,
            )
        except Exception as e:
            dur = (time.perf_counter() - start) * 1000
            return ToolOutput(
                tool_name=self.name,
                input=input_str,
                output=f"Error: {e}",
                success=False,
                duration_ms=dur,
            )
