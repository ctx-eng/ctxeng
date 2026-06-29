from __future__ import annotations

from typing import List, Optional

from ctxeng.assembly.assembler import ContextAssembler
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
    assembler = ContextAssembler(store=store)
    tracer = ContextTracer(assembler)
    turns: List[ConversationTurn] = []
    last_trace: Optional[ContextTrace] = None

    print("CtxEng CLI — type your messages. Commands: /help, /memories, /trace, /eval, /exit")

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
            print("Commands:  /help  /memories  /trace  /eval  /exit")
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

        turns.append(ConversationTurn(role="user", content=line))
        store.add(user_id, line)

        prompt, last_trace = tracer.assemble(user_id, turns, line)
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
