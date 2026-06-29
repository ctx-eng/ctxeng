from __future__ import annotations

from ctxeng.assembly.assembler import ContextAssembler
from ctxeng.core.context_manager import ContextManager
from ctxeng.eval.benchmark import BenchmarkRunner, format_benchmark_result
from ctxeng.eval.datasets import BUILT_IN_DATASETS, list_datasets
from ctxeng.models import ConversationTurn
from ctxeng.observability.reporter import format_trace
from ctxeng.observability.schema import ContextTrace
from ctxeng.observability.tracer import ContextTracer
from ctxeng.retrieval.hybrid import HybridRetriever
from ctxeng.stores.memory import InMemoryStore


def run_cli() -> None:
    store = InMemoryStore()
    mgr = ContextManager(memory_store=store)
    assembler = ContextAssembler(store=store)
    tracer = ContextTracer(assembler)
    turns: list[ConversationTurn] = []
    last_trace: ContextTrace | None = None

    print("CtxEng CLI — type your messages. Commands: /help, /memories, /trace, /eval, /tools, /exit")

    user_id = "cli-user"

    while True:
        try:
            line = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        if line == "/exit":
            break
        elif line == "/help":
            print("Commands:  /help  /memories  /trace  /eval  /tools  /exit")
            continue
        elif line == "/trace":
            if last_trace:
                print(format_trace(last_trace))
            else:
                print("  (no trace yet)")
            continue
        elif line == "/eval":
            _run_eval(store)
            continue
        elif line == "/memories":
            memories = store.search(user_id, "")
            if memories:
                for m in memories:
                    print(f"  - {m.text}")
            else:
                print("  (no memories)")
            continue
        elif line == "/tools":
            names = mgr._tool_registry.list()
            print("Registered tools:")
            for name in names:
                print(f"  - {name}")
            print("Tip: mention a tool name in your query to auto-execute it.")
            continue

        turns.append(ConversationTurn(role="user", content=line))
        store.add(user_id, line)

        tool_outputs = mgr.detect_and_run_tools(line)
        if tool_outputs:
            print(f"\n{'─' * 30} Tool Results {'─' * 30}")
            for t in tool_outputs:
                status = "OK" if t.success else "FAIL"
                print(f"  [{status}] {t.tool_name}({t.input}): {t.output[:300]}")
            print(f"{'─' * 75}\n")

        prompt, last_trace = tracer.assemble(user_id, turns, line, tool_outputs=tool_outputs)
        print(f"\n{'─' * 50}")
        print(prompt)
        print(f"{'─' * 50}\n")


def _run_eval(store: InMemoryStore) -> None:
    print(f"Running eval on {len(BUILT_IN_DATASETS)} datasets...")
    retriever = HybridRetriever(store)
    assembler = ContextAssembler(store=store)
    runner = BenchmarkRunner(store, retriever, assembler)
    for name in list_datasets():
        dataset = BUILT_IN_DATASETS[name]
        result = runner.run_dataset(dataset)
        print()
        print(format_benchmark_result(result))
    print()


if __name__ == "__main__":
    run_cli()
