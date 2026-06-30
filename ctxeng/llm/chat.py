from __future__ import annotations

from ctxeng.assembly.assembler import ContextAssembler
from ctxeng.llm.base import LLMMessage, LLMProvider, LLMResponse
from ctxeng.models import ConversationTurn


def run_chat(
    provider: LLMProvider,
    user_id: str = "chat-user",
    system_instruction: str | None = None,
) -> None:
    from ctxeng.stores.memory import InMemoryStore

    store = InMemoryStore()
    assembler = ContextAssembler(store=store)
    turns: list[ConversationTurn] = []

    print(f"CtxEng Chat ({provider.model_name}) — /exit to quit\n")

    while True:
        try:
            line = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line or line == "/exit":
            break

        turns.append(ConversationTurn(role="user", content=line))
        store.add(user_id, line)

        prompt = assembler.assemble(user_id, turns, line)
        messages = [
            LLMMessage(role="system", content=prompt),
            LLMMessage(role="user", content=line),
        ]

        print("Assistant: ", end="", flush=True)
        response = provider.stream(messages)
        print(response.content)

        turns.append(ConversationTurn(role="assistant", content=response.content))


def generate_reply(
    provider: LLMProvider,
    prompt: str,
    user_message: str,
) -> LLMResponse:
    messages = [
        LLMMessage(role="system", content=prompt),
        LLMMessage(role="user", content=user_message),
    ]
    return provider.generate(messages)
