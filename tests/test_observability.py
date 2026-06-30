from __future__ import annotations

from ctxeng.assembly.assembler import ContextAssembler
from ctxeng.models import ConversationTurn
from ctxeng.observability.reporter import format_trace, format_trace_short
from ctxeng.observability.schema import ContextSpan, ContextTrace
from ctxeng.observability.tracer import ContextTracer, get_trace, list_traces
from ctxeng.stores.memory import InMemoryStore


class TestContextSpan:
    def test_summary_with_all_fields(self) -> None:
        span = ContextSpan(
            stage="retrieve",
            input={"query": "test"},
            output={"item_count": 5},
            duration_ms=12.3,
            token_count=150,
        )
        s = span.summary
        assert "retrieve" in s
        assert "5 items" in s
        assert "150 tokens" in s
        assert "12.3ms" in s

    def test_summary_minimal(self) -> None:
        span = ContextSpan(
            stage="assemble",
            input={},
            output={},
            duration_ms=1.0,
        )
        assert "assemble" in span.summary
        assert "1.0ms" in span.summary

    def test_id_generated(self) -> None:
        span = ContextSpan(stage="test", input={}, output={}, duration_ms=0)
        assert len(span.id) > 0

    def test_parent_id_optional(self) -> None:
        child = ContextSpan(
            stage="child", input={}, output={}, duration_ms=0, parent_id="parent123"
        )
        assert child.parent_id == "parent123"
        orphan = ContextSpan(stage="orphan", input={}, output={}, duration_ms=0)
        assert orphan.parent_id is None


class TestContextTrace:
    def test_add_span_updates_totals(self) -> None:
        trace = ContextTrace(user_id="alice", query="test")
        trace.add_span(ContextSpan(stage="a", input={}, output={}, duration_ms=10, token_count=50))
        trace.add_span(ContextSpan(stage="b", input={}, output={}, duration_ms=20, token_count=100))
        assert trace.total_duration_ms == 30
        assert trace.total_tokens == 150
        assert len(trace.spans) == 2

    def test_trace_id_generated(self) -> None:
        trace = ContextTrace(user_id="alice", query="test")
        assert len(trace.trace_id) > 0

    def test_prompt_stored(self) -> None:
        trace = ContextTrace(user_id="alice", query="test")
        trace.prompt = "Hello world"
        assert trace.prompt == "Hello world"


class TestContextTracer:
    def test_assemble_produces_trace(self) -> None:
        store = InMemoryStore()
        store.add("alice", "Alice likes concise answers.")
        assembler = ContextAssembler(store=store)
        tracer = ContextTracer(assembler)
        prompt, trace = tracer.assemble(
            "alice",
            [ConversationTurn(role="user", content="Hello")],
            "concise",
        )
        assert trace.trace_id is not None
        assert trace.user_id == "alice"
        assert trace.query == "concise"
        assert len(trace.spans) >= 4
        assert any(s.stage == "retrieve" for s in trace.spans)
        assert any(s.stage == "assemble" for s in trace.spans)

    def test_assemble_returns_valid_prompt(self) -> None:
        store = InMemoryStore()
        store.add("alice", "Alice likes concise answers.")
        assembler = ContextAssembler(store=store)
        tracer = ContextTracer(assembler)
        prompt, trace = tracer.assemble("alice", [], "test")
        assert "Alice likes concise answers." in prompt
        assert "test" in prompt
        assert trace.prompt == prompt

    def test_trace_contains_retrieve_details(self) -> None:
        store = InMemoryStore()
        store.add("alice", "Alice likes concise answers.")
        assembler = ContextAssembler(store=store)
        tracer = ContextTracer(assembler)
        _, trace = tracer.assemble("alice", [], "concise")
        retrieve_span = next(s for s in trace.spans if s.stage == "retrieve")
        assert retrieve_span.output.get("item_count", 0) >= 1
        assert "score" in retrieve_span.output.get("items", [{}])[0]

    def test_trace_unknown_user(self) -> None:
        store = InMemoryStore()
        store.add("alice", "Alice's memory")
        assembler = ContextAssembler(store=store)
        tracer = ContextTracer(assembler)
        _, trace = tracer.assemble("bob", [], "test")
        retrieve_span = next(s for s in trace.spans if s.stage == "retrieve")
        assert retrieve_span.output.get("item_count", 0) == 0

    def test_trace_stored_in_global_registry(self) -> None:
        from ctxeng.observability.tracer import _trace_store

        _trace_store.clear()
        store = InMemoryStore()
        assembler = ContextAssembler(store=store)
        tracer = ContextTracer(assembler)
        _, trace = tracer.assemble("alice", [], "test")
        retrieved = get_trace(trace.trace_id)
        assert retrieved is not None
        assert retrieved.trace_id == trace.trace_id

    def test_list_traces(self) -> None:
        from ctxeng.observability.tracer import _trace_store

        _trace_store.clear()
        store = InMemoryStore()
        assembler = ContextAssembler(store=store)
        tracer = ContextTracer(assembler)
        tracer.assemble("alice", [], "q1")
        tracer.assemble("bob", [], "q2")
        all_traces = list_traces()
        assert len(all_traces) == 2
        alice_traces = list_traces(user_id="alice")
        assert len(alice_traces) == 1

    def test_list_traces_limit(self) -> None:
        from ctxeng.observability.tracer import _trace_store

        _trace_store.clear()
        store = InMemoryStore()
        assembler = ContextAssembler(store=store)
        tracer = ContextTracer(assembler)
        for i in range(20):
            tracer.assemble("alice", [], f"q{i}")
        assert len(list_traces(limit=5)) == 5


class TestReporter:
    def test_format_trace_contains_trace_id(self) -> None:
        trace = ContextTrace(user_id="alice", query="hello")
        trace.add_span(ContextSpan(stage="test", input={}, output={}, duration_ms=1))
        output = format_trace(trace)
        assert trace.trace_id in output

    def test_format_trace_short_compact(self) -> None:
        trace = ContextTrace(user_id="alice", query="hello")
        trace.add_span(
            ContextSpan(stage="retrieve", input={}, output={"item_count": 3}, duration_ms=10)
        )
        short = format_trace_short(trace)
        assert trace.trace_id in short
        assert "retrieve" in short
