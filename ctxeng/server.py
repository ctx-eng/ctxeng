from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ctxeng.core.context_manager import ContextManager
from ctxeng.core.memory_store import MemoryStore
from ctxeng.models import ConversationTurn, MemoryItem

app = FastAPI(title="CtxEng API", version="0.1.0")

_store = MemoryStore()
_manager = ContextManager(memory_store=_store)


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
    id: int
    user_id: str
    text: str
    timestamp: Optional[str] = None


class PromptResponse(BaseModel):
    prompt: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/memories", status_code=201)
def add_memory(body: AddMemoryRequest) -> MemoryResponse:
    memory = _store.add_memory(body.user_id, body.text)
    return MemoryResponse(
        id=id(memory),
        user_id=memory.user_id,
        text=memory.text,
        timestamp=memory.timestamp,
    )


@app.get("/memories/{user_id}")
def search_memories(user_id: str, query: str = "") -> List[MemoryResponse]:
    results = _store.search(user_id, query) if query else [
        m for m in _store._memories if m.user_id == user_id
    ]
    return [
        MemoryResponse(
            id=id(m),
            user_id=m.user_id,
            text=m.text,
            timestamp=m.timestamp,
        )
        for m in results
    ]


@app.post("/prompt", response_model=PromptResponse)
def build_prompt(body: BuildPromptRequest) -> PromptResponse:
    try:
        prompt = _manager.build_prompt(body.user_id, body.turns, body.current_query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return PromptResponse(prompt=prompt)
