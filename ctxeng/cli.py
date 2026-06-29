from __future__ import annotations

from typing import List, Optional

from ctxeng.assembly.assembler import ContextAssembler
from ctxeng.models import ConversationTurn
from ctxeng.observability.reporter import format_trace
from ctxeng.observability.schema import ContextTrace
from ctxeng.observability.tracer import ContextTracer
from ctxeng.stores.memory import InMemoryStore


def run_cli() -> None:
    store = InMemoryStore()
    assembler = ContextAssembler(store=store)
    tracer = ContextTracer(assembler)
    turns: List[ConversationTurn] = []
    last_trace: Optional[ContextTrace] = None

    print("CtxEng CLI — type your messages. Commands: /help, /memories, /trace, /exit")

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
            print("Commands:  /help  /memories  /trace  /exit")
            continue
        elif line == "/trace":
            if last_trace:
                print(format_trace(last_trace))
            else:
                print("  (no trace yet)")
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


if __name__ == "__main__":
    run_cli()
