from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ctxeng.core.context_manager import ContextManager
from ctxeng.models import ConversationTurn, MemoryItem
from ctxeng.stores.memory import InMemoryStore

app = FastAPI(title="CtxEng API", version="0.1.0")

_store = InMemoryStore()
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
    id: str
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
        prompt = _manager.build_prompt(body.user_id, body.turns, body.current_query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return PromptResponse(prompt=prompt)
