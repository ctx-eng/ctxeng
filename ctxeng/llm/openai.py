from __future__ import annotations

from typing import List, Optional

from ctxeng.llm.base import LLMMessage, LLMProvider, LLMResponse


class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai SDK is required. Install with: pip install openai"
            )
        self._model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        raw = [{"role": m.role, "content": m.content} for m in messages]
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=raw,
            **kwargs,
        )
        choice = resp.choices[0]
        return LLMResponse(
            content=choice.message.content or "",
            finish_reason=choice.finish_reason or "stop",
            usage={
                "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
            },
        )

    def stream(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        raw = [{"role": m.role, "content": m.content} for m in messages]
        collected: List[str] = []
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=raw,
            stream=True,
            **{k: v for k, v in kwargs.items() if k != "stream"},
        )
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                collected.append(delta.content)
        return LLMResponse(
            content="".join(collected),
            finish_reason="stop",
        )
