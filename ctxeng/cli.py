from __future__ import annotations

from typing import List

from ctxeng.core.context_manager import ContextManager
from ctxeng.models import ConversationTurn
from ctxeng.stores.memory import InMemoryStore


def run_cli() -> None:
    store = InMemoryStore()
    manager = ContextManager(memory_store=store)
    turns: List[ConversationTurn] = []

    print("CtxEng CLI — type your messages. Commands: /help, /memories, /exit")

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
            print("Commands:  /help  /memories  /exit")
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

        prompt = manager.build_prompt(user_id, turns, line)
        print(f"\n{'─' * 50}")
        print(prompt)
        print(f"{'─' * 50}\n")


if __name__ == "__main__":
    run_cli()
