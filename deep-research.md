# ContextOS — Next-Gen Context Engineering Framework

## Executive Summary

**ContextOS** is not another LLM wrapper. It is a **context operating system** — a modular, observable, evaluation-driven pipeline that ingests, processes, stores, retrieves, assembles, compresses, and injects context into any LLM call. It is designed from the ground up to be model-agnostic, storage-pluggable, and benchmark-verified.

The project lives at **`/Users/devenv/work/CtxEng`** and is currently an MVP scaffold with:
- In-memory `MemoryStore` (substring search)
- `ContextManager` prompt assembler
- CLI chat loop
- FastAPI server (`/health`, `/memories`, `/prompt`)
- 5 passing tests

This document describes the transformation from MVP into a next-generation context engineering framework.

---

## Core Thesis

> Context engineering is the discipline of controlling **what** an LLM sees, **when** it sees it, and **how** it's presented — all within a finite token budget. ContextOS makes this systematic, observable, and measurable.

### Design Principles

| Principle | Meaning |
|-----------|---------|
| **Context-first, not LLM-first** | The pipeline is independent of any specific LLM. Swap models without changing context logic. |
| **Observable by default** | Every retrieval decision, ranking score, and token allocation is traceable. No black boxes. |
| **Evaluation-driven** | Every retrieval strategy is benchmarked before it ships. Precision, recall, and token efficiency are tracked. |
| **Pluggable storage** | Start with in-memory, graduate to SQLite for persistence, then vector DB for scale — all through a common ABC. |
| **Token-budget-aware** | The assembler never exceeds its budget. It prioritizes, compresses, and truncates intelligently. |
| **Minimal core, rich extras** | Core dependencies stay lean. Heavy features (embeddings, vector DBs) are optional extras. |

---

## Architecture: The Context Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CONTEXT PIPELINE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input ──► Ingest ──► Process ──► Store ──► Retrieve ──► Assemble  │
│  (query,     (raw        (chunk,     (vector,    (semantic,    (rank,    │
│   user_id,    text,       embed,      graph,      keyword,      dedup,   │
│   metadata)   files,      tag,        key-value)  temporal,     prior-   │
│               structured  filter PII)             structural)   itize)  │
│               data)                                                        │
│                                                                     │
│  ──► Compress ──► Inject ──► Output                                  │
│      (summarize,   (format        (enriched                            │
│       truncate     into system/   prompt)                              │
│       to token     user                                                │
│       budget)     messages)                                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
                     ┌─────────────────┐
                     │  Observability  │
                     │  (Context       │
                     │   Tracing)      │
                     └─────────────────┘
```

### Pipeline Stage Details

| Stage | Responsibility | Key Questions |
|-------|---------------|---------------|
| **Ingest** | Accept raw input — text, files, structured data, images | What format is the source? What metadata accompanies it? |
| **Process** | Chunk, embed, tag, extract entities, filter PII | How big should chunks be? Which embedding model? What tags? |
| **Store** | Persist to pluggable backends | Vector for similarity? Graph for relations? KV for fast lookup? |
| **Retrieve** | Fetch relevant context via hybrid search | Semantic matches? Keyword fallback? Temporal relevance? Structural links? |
| **Assemble** | Rank, deduplicate, prioritize, slot into templates | How many items fit in the budget? Which are most relevant? |
| **Compress** | Summarize or truncate to stay under token limit | Should we summarize the top 5 or include 10 truncated? |
| **Inject** | Format into system message, user message, or tool schema | What's the output format for this LLM? |

---

## Memory Architecture: Three-Tier Hierarchy

```
┌──────────────────────────────────────────────────┐
│                 WORKING MEMORY                    │
│  (current session, ~5-20 recent turns)           │
│  Volatile, full-fidelity, auto-expired           │
├──────────────────────────────────────────────────┤
│                 EPISODIC MEMORY                   │
│  (recent sessions, compressed summaries)          │
│  Persisted, summarized, consolidated periodically │
├──────────────────────────────────────────────────┤
│                 LONG-TERM MEMORY                  │
│  (cross-session patterns, user preferences,       │
│   factual knowledge)                              │
│  Embedded + indexed, semantic retrieval           │
└──────────────────────────────────────────────────┘
```

| Tier | Backend | Retention | Access Pattern |
|------|---------|-----------|----------------|
| Working | In-memory (Python list) | Session-only | Full scan |
| Episodic | SQLite | Days-weeks | Keyword + temporal |
| Long-term | Vector DB (chromadb/lancedb) | Indefinite | Semantic + hybrid |

**Consolidation policy:** Every N turns (configurable), the oldest working memory entries are summarized into episodic memory. Every M sessions, episodic summaries are consolidated into long-term embeddings.

---

## Implementation Phases

### Phase 1: Memory Tiering & Pluggable Stores

**Goal:** Replace the monolithic `MemoryStore` with a `BaseStore` ABC and three implementations (in-memory, SQLite, vector stub).

**Files to create/modify:**

| File | What it contains |
|------|-----------------|
| `ctxeng/stores/__init__.py` | Package init |
| `ctxeng/stores/base.py` | `ContextStore` ABC with `add`, `search`, `delete`, `list` |
| `ctxeng/stores/memory.py` | `InMemoryStore` — refactored from `memory_store.py` |
| `ctxeng/stores/sqlite.py` | `SQLiteStore` — persistent, thread-safe, indexed |
| `ctxeng/stores/vector.py` | `VectorStore` — chromadb/lancedb wrapper (optional dep) |
| `ctxeng/core/memory_store.py` | Remove or re-export from stores |
| `tests/test_stores.py` | Tests for all three store backends |

**Key decisions:**
- ABC uses `MemoryItem` as the universal record type
- `search()` returns `List[MemoryItem]` with a relevance `score` field
- Stores are composable: a `TieredStore` can query working → episodic → long-term in order

**Acceptance criteria:**
- All three stores pass the same test suite (via ABC conformance)
- SQLite store persists across process restarts
- `pytest -q` passes (existing + new tests)

---

### Phase 2: Semantic Retrieval & Hybrid Search

**Goal:** Replace substring search with embedding-based semantic search, falling back to keyword + substring.

**Files to create/modify:**

| File | What it contains |
|------|-----------------|
| `ctxeng/retrieval/__init__.py` | Package init |
| `ctxeng/retrieval/embeddings.py` | `EmbeddingModel` wrapper (sentence-transformers) |
| `ctxeng/retrieval/hybrid.py` | `HybridRetriever` — dense + sparse (BM25) + keyword |
| `ctxeng/retrieval/reranker.py` | `CrossEncoderReranker` — re-rank top-k results |
| `tests/test_retrieval.py` | Tests for retrieval pipeline |

**Key decisions:**
- Embedding model is configurable, default `all-MiniLM-L6-v2` (384 dim, fast)
- BM25 via `rank_bm25` library (pure Python, no heavy deps)
- Hybrid score = `α * dense_score + (1-α) * sparse_score` (configurable α)
- Re-ranking is optional (cross-encoder is expensive — opt-in)
- All embedding deps are extras: `ctxeng[vector]`

**Acceptance criteria:**
- `HybridRetriever.search("user", "concise")` returns semantically related results even without exact substring match
- BM25 fallback works without embeddings installed
- Configurable top-k, α, and re-ranking enable/disable

---

### Phase 3: Context Assembly Engine

**Goal:** Build a token-budget-aware assembler that replaces the current naive `build_prompt`.

**Files to create/modify:**

| File | What it contains |
|------|-----------------|
| `ctxeng/assembly/__init__.py` | Package init |
| `ctxeng/assembly/assembler.py` | `ContextAssembler` — budget-aware prioritization |
| `ctxeng/assembly/templates.py` | `PromptTemplate` registry with slot-filling |
| `ctxeng/assembly/prioritizer.py` | Relevance scoring, deduplication, diversity |
| `tests/test_assembly.py` | Tests for assembly logic |

**Key features:**
- **Token budget:** Assembler takes a `max_tokens` param and allocates across memory items, history, and current query
- **Prioritization:** Items are sorted by relevance score, then by recency, then by diversity (MMR-style)
- **Deduplication:** Near-duplicate items (cosine > 0.95) are collapsed
- **Templates:** System prompt is a Jinja2 template with slots for `{memories}`, `{history}`, `{query}`, `{instructions}`
- **Fallback:** If budget is exceeded, items are first summarized, then truncated with `...`

**Acceptance criteria:**
- `Assembler.assemble(query, memories, history, max_tokens=2048)` never exceeds budget
- Prioritization respects relevance score ordering
- Templates are overridable by the user
- Deduplication removes near-identical memories

---

### Phase 4: Observability (Context Tracing)

**Goal:** Make every context decision traceable, debuggable, and introspectable.

**Files to create/modify:**

| File | What it contains |
|------|-----------------|
| `ctxeng/observability/__init__.py` | Package init |
| `ctxeng/observability/tracer.py` | `ContextTracer` — span-based tracing |
| `ctxeng/observability/reporter.py` | Human-readable trace formatter |
| `ctxeng/observability/schema.py` | Trace data models |
| `ctxeng/cli.py` | Add `/trace`, `/context`, `/explain` commands |
| `ctxeng/server.py` | Add `POST /context/explain`, `GET /context/trace/{id}` |
| `tests/test_observability.py` | Tests for tracing |

**Trace data model:**

```python
@dataclass
class ContextSpan:
    id: str
    parent_id: Optional[str]
    stage: str  # "retrieve" | "assemble" | "compress" | "inject"
    input: dict
    output: dict
    duration_ms: float
    token_count: int
    timestamp: str

@dataclass
class ContextTrace:
    trace_id: str
    user_id: str
    query: str
    spans: List[ContextSpan]
    total_duration_ms: float
    total_tokens: int
```

**CLI introspection:**

```
You: /context
Context trace #abc123:
  retrieve: 3 memories found (2 from working, 1 from long-term)
    - "Alice likes concise answers" (score: 0.89)
    - "Alice prefers bullet points" (score: 0.72)
    - "Alice mentioned Python 3.12" (score: 0.31, from long-term)
  assemble: 2 items included (budget 512 tokens, used 124)
  compress: no compression needed
  inject: formatted into template "default"
```

**Acceptance criteria:**
- Every `ContextManager.build_prompt()` call produces a `ContextTrace`
- Traces are accessible via `/trace` CLI command
- Traces are accessible via REST API
- Trace includes per-stage timing and token counts

---

### Phase 5: Context Compression

**Goal:** Automatically compress long histories and consolidate memories across sessions.

**Files to create/modify:**

| File | What it contains |
|------|-----------------|
| `ctxeng/compression/__init__.py` | Package init |
| `ctxeng/compression/summarizer.py` | `ContextSummarizer` — map-reduce summarization |
| `ctxeng/compression/consolidator.py` | `MemoryConsolidator` — working → episodic → long-term |
| `tests/test_compression.py` | Tests for compression |

**Summarization strategies:**
1. **Sliding window:** Keep last N turns, summarize everything older
2. **Map-reduce:** Chunk conversations, summarize each chunk, reduce into a single summary
3. **Recursive:** If summary is still too long, summarize the summary

**Consolidation policy:**
- Every 10 turns (configurable), oldest 5 working memory items → one episodic summary
- Every 5 sessions (configurable), episodic summaries → embedded long-term memory
- Episodic summaries include: `(date_range, topics, key_facts, user_preferences)`

**Acceptance criteria:**
- Summarizer produces a coherent summary given any list of turns
- Consolidator respects configurable thresholds
- Summarized context is still useful for retrieval (tested via Phase 6 eval)

---

### Phase 6: Evaluation Harness

**Goal:** Measure retrieval quality, token efficiency, and overall context usefulness — automatically.

**Files to create/modify:**

| File | What it contains |
|------|-----------------|
| `ctxeng/eval/__init__.py` | Package init |
| `ctxeng/eval/metrics.py` | `ContextMetrics` — precision, recall, MRR, token efficiency |
| `ctxeng/eval/benchmark.py` | `BenchmarkRunner` — automated evaluation suite |
| `ctxeng/eval/datasets.py` | Built-in eval datasets + dataset loader |
| `ctxeng/cli.py` | Add `/eval` command |
| `tests/test_eval.py` | Tests for evaluation metrics |

**Metrics:**

| Metric | Definition |
|--------|------------|
| **Precision@k** | Fraction of retrieved items that are relevant |
| **Recall@k** | Fraction of relevant items that are retrieved |
| **MRR** | Mean reciprocal rank of first relevant item |
| **Token efficiency** | (Relevant tokens) / (Total tokens in prompt) |
| **Budget utilization** | (Actual tokens) / (Max budget) |
| **Latency p50/p95/p99** | Retrieval and assembly latency |

**Built-in datasets:**
- `simple_preferences`: 20 user preference statements with queries
- `multi_topic_conversations`: 10 multi-turn conversations with topic shifts
- `long_term_cross_session`: 5 users, 3 sessions each, cross-session queries

**Acceptance criteria:**
- `python -m ctxeng.eval benchmark` runs all datasets and prints a report
- CLI `/eval` runs a quick sanity check
- Metrics are comparable across retrieval strategies

---

### Phase 7: Multi-Modal & Structured Context

**Goal:** Support context from images (caption-based), structured data (SQL, JSON, CSV), and files.

**Files to create/modify:**

| File | What it contains |
|------|-----------------|
| `ctxeng/ingestion/__init__.py` | Package init |
| `ctxeng/ingestion/text.py` | Text file ingestion (txt, md, pdf) |
| `ctxeng/ingestion/image.py` | Image ingestion — caption via `transformers` or `ollama` |
| `ctxeng/ingestion/structured.py` | Structured data ingestion (CSV, JSON, SQL results) |
| `tests/test_ingestion.py` | Tests for ingestion |

**Key decisions:**
- Images are converted to text captions, stored as regular memories
- Structured data is converted to natural language descriptions
- File type detection via `python-magic` or extension
- Heavy dependencies (transformers, torch) are extras only

**Acceptance criteria:**
- Image → caption → memory pipeline works end-to-end
- CSV/JSON → description → memory works end-to-end
- All ingestors produce `MemoryItem` instances

---

### Phase 8: Agentic Context Router

**Goal:** Support context that spans multiple tool calls, agent steps, and branching conversations.

**Files to create/modify:**

| File | What it contains |
|------|-----------------|
| `ctxeng/routing/__init__.py` | Package init |
| `ctxeng/routing/router.py` | `ContextRouter` — multi-step context propagation |
| `ctxeng/routing/diff.py` | Context diff between turns (what changed, what persisted) |
| `ctxeng/routing/lifecycle.py` | Context lifecycle manager |
| `tests/test_routing.py` | Tests for context routing |

**Key features:**
- **Context diff:** After each agent step, compute what context was used and what new context was produced
- **Branching:** If the agent forks into parallel paths, each branch gets its own context slice
- **Lifecycle:** Context is born (created), lives (updated each turn), and dies (expired or explicitly destroyed)
- **Propagation:** Relevant context from step N is automatically carried to step N+1

**Acceptance criteria:**
- Agent with 3 sequential tool calls gets correct context at each step
- Context diff correctly identifies added/removed/changed items
- Branching conversations maintain independent context per branch

---

## File Map (Complete)

```
ctxeng/
├── __init__.py
├── app.py                          # Demo entry point (unchanged)
├── cli.py                          # Extended with /trace, /context, /eval
├── models.py                       # Extended with ContextSpan, ContextTrace
├── server.py                       # Extended with observability endpoints
├── core/
│   ├── __init__.py
│   ├── context_manager.py          # Refactored to use assembly engine
│   └── memory_store.py             # Deprecated → re-exports from stores/
├── stores/
│   ├── __init__.py
│   ├── base.py                     # ContextStore ABC
│   ├── memory.py                   # InMemoryStore
│   ├── sqlite.py                   # SQLiteStore
│   └── vector.py                   # VectorStore (optional)
├── retrieval/
│   ├── __init__.py
│   ├── embeddings.py               # EmbeddingModel wrapper
│   ├── hybrid.py                   # HybridRetriever
│   └── reranker.py                 # CrossEncoderReranker
├── assembly/
│   ├── __init__.py
│   ├── assembler.py                # ContextAssembler
│   ├── templates.py                # PromptTemplate registry
│   └── prioritizer.py              # Relevance scoring, dedup
├── compression/
│   ├── __init__.py
│   ├── summarizer.py               # ContextSummarizer
│   └── consolidator.py             # MemoryConsolidator
├── observability/
│   ├── __init__.py
│   ├── tracer.py                   # ContextTracer
│   ├── reporter.py                 # Trace formatter
│   └── schema.py                   # Trace data models
├── ingestion/
│   ├── __init__.py
│   ├── text.py                     # Text file ingestion
│   ├── image.py                    # Image caption ingestion
│   └── structured.py               # Structured data ingestion
├── routing/
│   ├── __init__.py
│   ├── router.py                   # ContextRouter
│   ├── diff.py                     # Context diff
│   └── lifecycle.py                # Context lifecycle
└── eval/
    ├── __init__.py
    ├── metrics.py                  # ContextMetrics
    ├── benchmark.py                # BenchmarkRunner
    └── datasets.py                 # Eval datasets

tests/
├── __init__.py
├── test_context_manager.py         # (unchanged)
├── test_server.py                  # Extended
├── test_stores.py
├── test_retrieval.py
├── test_assembly.py
├── test_compression.py
├── test_observability.py
├── test_ingestion.py
├── test_routing.py
└── test_eval.py
```

---

## Dependency Strategy

```toml
[project]
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.34",
]

[project.optional-dependencies]
# Lightweight extras
sqlite = []  # sqlite3 is stdlib
retrieval = [
    "rank-bm25>=0.2",
]
templates = [
    "jinja2>=3.1",
]

# Heavy extras (install only when needed)
vector = [
    "sentence-transformers>=3.0",
    "chromadb>=0.5",
    "torch>=2.0",  # required by sentence-transformers
]
ingestion = [
    "python-magic>=0.4",
    "pillow>=10.0",
    "transformers>=4.40",  # for image captioning
]
all = [
    "ctxeng[retrieval, templates, vector, ingestion]",
]

[project.optional-dependencies]
test = ["httpx>=0.28"]
```

**Rationale:**
- Core (fastapi, uvicorn) stays tiny — ~10 MB
- Retrieval extras (rank-bm25, jinja2) are still lightweight — ~5 MB
- Vector extras (sentence-transformers, torch) are large — ~1.5 GB — clearly opt-in
- Ingestion extras (transformers, pillow) are moderate — ~200 MB
- Developer can start with `pip install ctxeng` and never touch ML deps

---

## Evaluation-Driven Development Workflow

```
1. Define a benchmark dataset
2. Implement a retrieval strategy
3. Run benchmark against the dataset
4. Compare metrics (precision, recall, token efficiency)
5. If metrics regress, fix before merging
6. Store benchmark results in git (JSON file)
```

**CLI usage:**
```bash
python -m ctxeng.eval benchmark                    # Run all datasets
python -m ctxeng.eval benchmark --dataset simple   # One dataset
python -m ctxeng.eval compare --baseline main      # Compare to git baseline
```

**CI integration:**
```yaml
# Future .github/workflows/eval.yml
- run: python -m ctxeng.eval benchmark --json results.json
- run: python -m ctxeng.eval check --json results.json --threshold precision=0.8
```

---

## Risk & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Embedding quality degrades for niche domains | Poor retrieval | Retrieval is hybrid (dense + sparse); BM25 catches exact terms |
| Token budget management is fragile | Assembler exceeds budget or wastes capacity | Strict allocation algorithm with hard cap; test coverage for edge cases |
| SQLite becomes bottleneck at scale | High latency | Pluggable backend — swap to PostgreSQL or Redis when needed |
| Evaluation datasets don't reflect real usage | Metrics are meaningless | Dataset format is open; users contribute their own datasets |
| LLM ecosystem moves too fast | Architecture becomes obsolete | Model-agnostic from the start; pipeline adapts to new APIs via template system |

---

## Timeline (Estimated)

| Phase | Effort | Dependencies |
|-------|--------|--------------|
| **P1: Memory Tiering** | 3-4 days | None |
| **P2: Semantic Retrieval** | 5-7 days | P1 (stores exist) |
| **P3: Assembly Engine** | 4-5 days | None (can work in parallel with P2) |
| **P4: Observability** | 3-4 days | P3 (assembly is traceable) |
| **P5: Compression** | 4-5 days | P3 (assembly produces measurable context) |
| **P6: Evaluation** | 4-5 days | P2, P3 (need retrieval + assembly to evaluate) |
| **P7: Multi-Modal** | 5-6 days | P1 (stores support MemoryItem) |
| **P8: Agentic Routing** | 4-5 days | P4 (tracing needed for diff) |

**Total: 8 phases, ~32-41 days of focused effort**

Phases can overlap: P2 + P3 can be built in parallel once P1 lands. P4 depends on P3 but not on P2. P6 should come after P2 and P3 are stable.

---

## Non-Goals

- **Not an LLM provider** — ContextOS does not ship with or require a specific LLM
- **Not a vector database** — It wraps existing vector DBs (chromadb, lancedb) via a common ABC
- **Not a chat framework** — It produces context; how that context is used is the caller's responsibility
- **Not a data pipeline** — It focuses on *context for LLMs*, not general ETL or data processing
- **Not a Kubernetes operator** — It is a Python library, not an infrastructure platform

---

## Comparison: ContextOS vs. Existing Approaches

| Aspect | Naive (MVP) | LangChain | ContextOS |
|--------|-------------|-----------|-----------|
| Memory | In-memory list | VectorStore abstraction | Three-tier with pluggable backends |
| Retrieval | Substring match | Embedding + vector DB | Hybrid (dense + sparse + keyword) + re-ranking |
| Assembly | String concat | PromptTemplate | Token-budget-aware prioritization + dedup |
| Observability | None | Callback handlers | Trace spans with structured schema |
| Evaluation | Manual | LangSmith (external) | Built-in benchmark runner |
| Compression | None | ConversationSummaryMemory | Map-reduce + configurable consolidation |
| Multi-modal | None | Depends on model | Ingest → caption → memory pipeline |
| Agentic | None | AgentExecutor | Context diff + lifecycle management |
| Dependencies | fastapi, uvicorn | 50+ transitive | Minimal core, heavy extras opt-in |

---

## Getting Started After Implementation

```bash
# Install core (lightweight)
pip install ctxeng

# Install with vector support
pip install "ctxeng[vector]"

# Run the CLI
python -m ctxeng.cli

# Run the server
uvicorn ctxeng.server:app

# Run evaluation
python -m ctxeng.eval benchmark

# Use as a library
from ctxeng.stores import SQLiteStore
from ctxeng.retrieval import HybridRetriever
from ctxeng.assembly import ContextAssembler

store = SQLiteStore(":memory:")
retriever = HybridRetriever(store)
assembler = ContextAssembler(max_tokens=2048)
prompt = assembler.assemble("user", "Tell me about Python", history=[])
```
