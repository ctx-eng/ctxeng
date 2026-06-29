from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from ctxeng.assembly.assembler import ContextAssembler
from ctxeng.llm.base import LLMMessage
from ctxeng.llm.chat import generate_reply
from ctxeng.llm.openai import OpenAIProvider
from ctxeng.models import ConversationTurn, MemoryItem
from ctxeng.observability.reporter import format_trace
from ctxeng.observability.tracer import ContextTracer, get_trace
from ctxeng.stores.memory import InMemoryStore

app = FastAPI(title="CtxEng API", version="0.1.0")

_store = InMemoryStore()
_assembler = ContextAssembler(store=_store)
_tracer = ContextTracer(_assembler)
_llm: Optional[OpenAIProvider] = None


def _get_llm() -> OpenAIProvider:
    global _llm
    if _llm is None:
        _llm = OpenAIProvider()
    return _llm


class AddMemoryRequest(BaseModel):
    user_id: str
    text: str


class SearchMemoryRequest(BaseModel):
    query: str


class BuildPromptRequest(BaseModel):
    user_id: str
    turns: List[ConversationTurn]
    current_query: str


class MemoryResponse(BaseModel):
    id: str
    user_id: str
    text: str
    timestamp: Optional[str] = None


class PromptResponse(BaseModel):
    prompt: str


class ExplainResponse(BaseModel):
    prompt: str
    trace_id: str
    trace_report: str


class TraceResponse(BaseModel):
    trace_id: str
    user_id: str
    query: str
    prompt: str
    total_duration_ms: float
    total_tokens: int


class ChatRequest(BaseModel):
    messages: List[LLMMessage]


class ChatResponse(BaseModel):
    content: str
    finish_reason: str = "stop"


CHAT_HTML = (Path(__file__).resolve().parent / "static" / "chat.html").read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def chat_ui():
    return HTMLResponse(CHAT_HTML)


@app.post("/memories", status_code=201)
def add_memory(body: AddMemoryRequest) -> MemoryResponse:
    memory = _store.add(body.user_id, body.text)
    return MemoryResponse(
        id=memory.id,
        user_id=memory.user_id,
        text=memory.text,
        timestamp=memory.timestamp,
    )


@app.get("/memories/{user_id}")
def search_memories(user_id: str, query: str = "") -> List[MemoryResponse]:
    results = _store.search(user_id, query) if query else _store.list(user_id)
    return [
        MemoryResponse(
            id=m.id,
            user_id=m.user_id,
            text=m.text,
            timestamp=m.timestamp,
        )
        for m in results
    ]


@app.post("/prompt", response_model=PromptResponse)
def build_prompt(body: BuildPromptRequest) -> PromptResponse:
    try:
        prompt, _ = _tracer.assemble(body.user_id, body.turns, body.current_query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return PromptResponse(prompt=prompt)


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest) -> ChatResponse:
    try:
        msgs = body.messages
        if not msgs:
            raise HTTPException(status_code=400, detail="No messages")
        system_msg = next((m for m in msgs if m.role == "system"), None)
        user_msg = next((m for m in msgs if m.role == "user"), None)
        if not system_msg or not user_msg:
            raise HTTPException(status_code=400, detail="Need system and user messages")
        resp = generate_reply(_get_llm(), system_msg.content, user_msg.content)
        return ChatResponse(content=resp.content, finish_reason=resp.finish_reason)
    except ImportError:
        raise HTTPException(status_code=503, detail="LLM provider not available (install openai)")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/context/explain", response_model=ExplainResponse)
def explain_prompt(body: BuildPromptRequest) -> ExplainResponse:
    try:
        prompt, trace = _tracer.assemble(body.user_id, body.turns, body.current_query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return ExplainResponse(
        prompt=prompt,
        trace_id=trace.trace_id,
        trace_report=format_trace(trace),
    )


@app.get("/context/trace/{trace_id}")
def get_context_trace(trace_id: str) -> TraceResponse:
    trace = get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return TraceResponse(
        trace_id=trace.trace_id,
        user_id=trace.user_id,
        query=trace.query,
        prompt=trace.prompt,
        total_duration_ms=trace.total_duration_ms,
        total_tokens=trace.total_tokens,
    )
