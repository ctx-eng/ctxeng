from __future__ import annotations

import json
import urllib.request
from typing import Any, Dict, List, Optional

from ctxeng.llm.base import LLMMessage, LLMProvider, LLMResponse


class OllamaProvider(LLMProvider):
    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434") -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")

    @property
    def model_name(self) -> str:
        return self._model

    def _post(self, payload: Dict[str, Any]) -> Any:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self._base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())

    def generate(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        raw = [{"role": m.role, "content": m.content} for m in messages]
        result = self._post({
            "model": self._model,
            "messages": raw,
            "stream": False,
            **kwargs,
        })
        return LLMResponse(
            content=result["message"]["content"],
            finish_reason=result.get("done_reason", "stop"),
        )

    def stream(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        raw = [{"role": m.role, "content": m.content} for m in messages]
        data = json.dumps({"model": self._model, "messages": raw, "stream": True, **kwargs}).encode()
        req = urllib.request.Request(
            f"{self._base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        collected: List[str] = []
        with urllib.request.urlopen(req) as resp:
            for line in resp:
                chunk = json.loads(line)
                if chunk.get("done"):
                    break
                collected.append(chunk.get("message", {}).get("content", ""))
        return LLMResponse(content="".join(collected))
