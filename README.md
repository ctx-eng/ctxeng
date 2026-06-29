# CtxEng

A lightweight context engineering framework — stores, retrieves, assembles, and
injects context into any LLM call.

```
pip install ctxeng
```

---

## Quickstart

```python
from ctxeng.stores.memory import InMemoryStore
from ctxeng.retrieval.hybrid import HybridRetriever
from ctxeng.assembly.assembler import ContextAssembler

store = InMemoryStore()
store.add("alice", "Alice prefers concise answers.")
store.add("alice", "Alice likes bullet points.")

retriever = HybridRetriever(store)
assembler = ContextAssembler(store=store, retriever=retriever, max_tokens=2048)

prompt = assembler.assemble(
    "alice",
    turns=[],
    query="How should I format the response?",
)
print(prompt)
```

---

## Architecture

```
Input (query + user_id)
  → Ingest (text, markdown, CSV, JSON, images)
    → Store (InMemory / SQLite / Vector)
      → Retrieve (hybrid dense+sparse+keyword)
        → Assemble (dedup, prioritize, template, trim to budget)
          → Trace (observability)
            → LLM (OpenAI / Ollama / custom)
```

CtxEng is **model-agnostic**. It produces enriched context — you plug in your own LLM.

---

## Features

### Stores (`ctxeng.stores`)

| Backend | Class | Persistence |
|---------|-------|-------------|
| In-memory | `InMemoryStore` | Ephemeral |
| SQLite | `SQLiteStore` | File-based (stdlib, no deps) |
| Vector (ChromaDB) | `VectorStore` | Ephemeral or persistent |

All stores conform to the `ContextStore` ABC (`add`, `search`, `delete`, `list`, `clear`).

```python
from ctxeng.stores.sqlite import SQLiteStore
store = SQLiteStore("ctxeng.db")
store.add("alice", "persistent memory")
```

### Retrieval (`ctxeng.retrieval`)

| Component | Description |
|-----------|-------------|
| `HybridRetriever` | Dense (sentence-transformers) + sparse (BM25) + keyword, configurable α |
| `CrossEncoderReranker` | Re-rank top-k with cross-encoder (optional) |

```python
retriever = HybridRetriever(store, alpha=0.6)
results = retriever.search("alice", "concise", top_k=5)
```

Heavy deps (`sentence-transformers`, `torch`) are opt-in:
```bash
pip install "ctxeng[vector]"
```

### Assembly (`ctxeng.assembly`)

| Component | Description |
|-----------|-------------|
| `ContextAssembler` | Retrieve → dedup → prioritize → template → trim to token budget |
| `Prioritizer` | Trigram dedup + MMR diversity ranking |
| `PromptTemplate` | Slot-based template (`{memories}`, `{history}`, `{query}`) |

```python
assembler = ContextAssembler(store=store, max_tokens=4096)
prompt = assembler.assemble("alice", turns, "What's the weather?")
```

### Observability (`ctxeng.observability`)

Every assembly produces a `ContextTrace` with per-stage spans:

```python
from ctxeng.observability.reporter import format_trace
prompt, trace = tracer.assemble("alice", turns, "query")
print(format_trace(trace))
```

```
Context trace #abc123
  [retrieve]    3 items, 0.7ms
  [deduplicate] removed 1 duplicate, 0.0ms
  [prioritize]  2 items, 0.0ms
  [assemble]    512 tokens, 1.2ms
Total: 1.9ms, 512 tokens, 4 stages
```

### Compression (`ctxeng.compression`)

| Component | Description |
|-----------|-------------|
| `ContextSummarizer` | Extractive TF-IDF summarization + sliding window |
| `MemoryConsolidator` | Working → episodic → long-term memory lifecycle |

```python
consolidator = MemoryConsolidator(store, turn_threshold=10, batch_size=5)
consolidator.record_turn("alice", "user message")
```

### Evaluation (`ctxeng.eval`)

```bash
python -m ctxeng.eval  # or /eval in CLI
```

| Dataset | Queries | Description |
|---------|---------|-------------|
| `simple_preferences` | 6 | User preference matching |
| `multi_topic_conversations` | 3 | Cross-topic retrieval |
| `long_term_cross_session` | 4 | Cross-session memory recall |

Metrics: P@1/3/5, R@3/5, MRR, MAP, token efficiency.

### Ingestion (`ctxeng.ingestion`)

```python
from ctxeng.ingestion.ingestor import FileIngestor
ingestor = FileIngestor()
memories = ingestor.ingest("document.md", user_id="alice")
```

Supported formats: `.txt`, `.md`, `.csv`, `.json`, `.jpg`, `.png`, `.py`, `.js`, and more.

### Routing (`ctxeng.routing`)

```python
router = ContextRouter(store)
r1 = router.step("alice", "What's my name?")
r2 = router.step("alice", "What did I just ask?")  # diff tracked
```

- `ContextDiff` — detects added, removed, and score-changed items between steps
- `LifecycleManager` — tracks born → active → stale → archived → dead per memory

### LLM Integration (`ctxeng.llm`)

```python
from ctxeng.llm.openai import OpenAIProvider
from ctxeng.llm.chat import generate_reply

provider = OpenAIProvider()  # uses OPENAI_API_KEY env var
resp = generate_reply(provider, prompt, "user question")
print(resp.content)
```

Or via the chat loop:

```bash
python -c "from ctxeng.llm.chat import run_chat; from ctxeng.llm.openai import OpenAIProvider; run_chat(OpenAIProvider())"
```

---

## CLI

```bash
python -m ctxeng.cli
```

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/memories` | List stored memories |
| `/trace` | Show last context assembly trace |
| `/eval` | Run benchmarks against built-in datasets |
| `/exit` | Quit |

---

## Server

```bash
uvicorn ctxeng.server:app
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Chat UI (HTML) |
| `/health` | GET | Health check |
| `/memories` | POST | Add a memory |
| `/memories/{user_id}` | GET | Search memories |
| `/prompt` | POST | Build a context-assembled prompt |
| `/chat` | POST | Prompt + LLM reply (requires `openai`) |
| `/context/explain` | POST | Prompt + trace report |
| `/context/trace/{id}` | GET | Retrieve stored trace |

---

## Optional Dependencies

| Extra | Packages |
|-------|----------|
| `templates` | `jinja2` |
| `retrieval` | `rank-bm25` |
| `vector` | `sentence-transformers`, `numpy`, `chromadb` |
| `llm` | `openai` |
| `all` | Everything above |

```bash
pip install "ctxeng[all]"
```

---

## Development

```bash
git clone <repo>
cd ctxeng
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all,test]"
pytest
```

---

## Project Structure

```
ctxeng/
├── models.py              # ConversationTurn, MemoryItem dataclasses
├── app.py                 # Demo entry point
├── cli.py                 # Interactive CLI
├── server.py              # FastAPI server
├── core/
│   └── context_manager.py # Legacy wrapper (delegates to ContextAssembler)
├── stores/                # Pluggable store backends
├── retrieval/             # Hybrid search, embeddings, re-ranking
├── assembly/              # Budget-aware context assembly
├── observability/         # Span-based tracing
├── compression/           # Summarization, memory consolidation
├── eval/                  # Evaluation harness, benchmarks
├── ingestion/             # Multi-modal file ingestion
├── routing/               # Agentic context routing, lifecycle
├── llm/                   # LLM provider abstraction
└── static/
    └── chat.html          # Web chat UI
```
