from __future__ import annotations

from ctxeng.core.context_manager import ContextManager
from ctxeng.models import ConversationTurn
from ctxeng.stores.memory import InMemoryStore
from ctxeng.tools.base import BaseTool, ToolOutput, ToolRegistry


def test_tool_output_defaults() -> None:
    o = ToolOutput(tool_name="test", input="foo", output="bar")
    assert o.success is True
    assert o.duration_ms == 0.0
    assert o.metadata == {}


def test_tool_output_failure() -> None:
    o = ToolOutput(tool_name="test", input="foo", output="err", success=False)
    assert o.success is False


def test_tool_registry_register_and_list() -> None:
    reg = ToolRegistry()
    assert reg.list() == []

    class Dummy(BaseTool):
        name = "dummy"
        description = "a test tool"

        def run(self, input_str: str) -> ToolOutput:
            return ToolOutput(tool_name=self.name, input=input_str, output="ok")

    reg.register(Dummy())
    assert reg.list() == ["dummy"]


def test_tool_registry_get() -> None:
    reg = ToolRegistry()

    class Dummy(BaseTool):
        name = "dummy"
        description = ""

        def run(self, input_str: str) -> ToolOutput:
            return ToolOutput(tool_name=self.name, input=input_str, output="ok")

    reg.register(Dummy())
    t = reg.get("dummy")
    assert t is not None
    assert t.name == "dummy"
    assert reg.get("nonexistent") is None


def test_tool_registry_execute() -> None:
    reg = ToolRegistry()

    class Dummy(BaseTool):
        name = "dummy"
        description = ""

        def run(self, input_str: str) -> ToolOutput:
            return ToolOutput(tool_name=self.name, input=input_str, output=f"got:{input_str}")

    reg.register(Dummy())
    result = reg.execute("dummy", "hello")
    assert result is not None
    assert result.output == "got:hello"
    assert reg.execute("nonexistent", "x") is None


def test_tool_registry_match() -> None:
    reg = ToolRegistry()

    class Calc(BaseTool):
        name = "calculator"
        description = ""

        def run(self, input_str: str) -> ToolOutput:
            return ToolOutput(tool_name=self.name, input=input_str, output="42")

    class Search(BaseTool):
        name = "web_search"
        description = ""

        def run(self, input_str: str) -> ToolOutput:
            return ToolOutput(tool_name=self.name, input=input_str, output="results")

    reg.register(Calc())
    reg.register(Search())

    matched = reg.match("use the calculator please")
    assert len(matched) == 1
    assert matched[0].name == "calculator"

    matched = reg.match("no match here")
    assert len(matched) == 0


def test_calculator_tool() -> None:
    reg = ToolRegistry()
    from ctxeng.tools.calculator import CalculatorTool

    reg.register(CalculatorTool())

    result = reg.execute("calculator", "2 + 3 * 4")
    assert result is not None
    assert result.output == "14"
    assert result.success is True

    result = reg.execute("calculator", "10 / 0")
    assert result is not None
    assert result.success is False

    result = reg.execute("calculator", "abc")
    assert result is not None
    assert result.success is False


def test_file_lookup_tool() -> None:
    store = InMemoryStore()
    store.add("u1", "my favorite color is blue")
    store.add("u1", "the capital of France is Paris")

    from ctxeng.tools.file_lookup import FileLookupTool

    ft = FileLookupTool(store=store, user_id="u1")
    result = ft.run("color")
    assert result.success is True
    assert "blue" in result.output

    result = ft.run("capital")
    assert result.success is True
    assert "Paris" in result.output


def test_file_lookup_tool_no_store() -> None:
    from ctxeng.tools.file_lookup import FileLookupTool

    ft = FileLookupTool()
    result = ft.run("anything")
    assert result.success is False
    assert "no memory store" in result.output


def test_web_search_tool_bad_url() -> None:
    from ctxeng.tools.web_search import WebSearchTool

    wt = WebSearchTool()
    result = wt.run("not a url")
    assert result.success is False
    assert "valid URL" in result.output


def test_context_manager_default_tools() -> None:
    mgr = ContextManager()
    tools = mgr._tool_registry.list()
    assert "calculator" in tools
    assert "web_search" in tools


def test_context_manager_detect_and_run() -> None:
    mgr = ContextManager()
    outputs = mgr.detect_and_run_tools("use calculator 5 + 3")
    assert len(outputs) > 0
    calc_out = [o for o in outputs if o.tool_name == "calculator"]
    assert len(calc_out) > 0
    assert calc_out[0].output == "8"


def test_context_manager_no_tool_match() -> None:
    mgr = ContextManager()
    outputs = mgr.detect_and_run_tools("hello, how are you?")
    assert outputs == []


def test_context_manager_set_file_lookup() -> None:
    mgr = ContextManager()
    mgr.set_file_lookup_store()
    assert "file_lookup" in mgr._tool_registry.list()


def test_assemble_with_tool_outputs() -> None:
    from ctxeng.assembly.assembler import ContextAssembler

    store = InMemoryStore()
    store.add("u1", "the sky is blue")
    assembler = ContextAssembler(store=store)
    turns = [ConversationTurn(role="user", content="hi")]
    tool_outs = [ToolOutput(tool_name="calculator", input="2+2", output="4")]

    prompt = assembler.assemble("u1", turns, "what color is the sky?", tool_outputs=tool_outs)
    assert "Tool outputs:" in prompt
    assert "calculator" in prompt
    assert "4" in prompt
    assert "Tool outputs" in prompt


def test_assemble_without_tools() -> None:
    from ctxeng.assembly.assembler import ContextAssembler

    store = InMemoryStore()
    store.add("u1", "the sky is blue")
    assembler = ContextAssembler(store=store)
    turns = [ConversationTurn(role="user", content="hi")]

    prompt = assembler.assemble("u1", turns, "what color?")
    assert "Tool outputs:" in prompt
    assert "- none" in prompt


def test_tool_to_openai_tool() -> None:
    from ctxeng.tools.calculator import CalculatorTool

    ct = CalculatorTool()
    schema = ct.to_openai_tool()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "calculator"
    assert "parameters" in schema["function"]
