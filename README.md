# CtxEng

A lightweight context engineering framework — stores, retrieves, assembles,
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
  → Safety (input validation, poisoning filter)
    → Ingest (text, markdown, CSV, JSON, images, PDF)
      → Store (InMemory / SQLite / Vector / Tiered)
        → Profile (user preferences, tags)
          → Tools (calculator, web search, file lookup)
            → Retrieve (hybrid dense+sparse+keyword + reranker)
              → Assemble (dedup, prioritize, template, trim to budget)
                → Compress (summarize, consolidate working→episodic→long-term)
                  → Trace (observability)
                    → LLM (OpenAI / Ollama / custom)
```

CtxEng is **model-agnostic** with **always-on observability** — you plug in your own LLM.

---

## Features

### Stores (`ctxeng.stores`)

| Backend | Class | Persistence |
|---------|-------|-------------|
| In-memory | `InMemoryStore` | Ephemeral |
| SQLite | `SQLiteStore` | File-based (stdlib, no deps) |
| Vector (ChromaDB) | `VectorStore` | Embedding-powered |
| Tiered | `TieredStore` | Composable three-tier fallback |

All stores conform to the `ContextStore` ABC (`add`, `search`, `delete`, `list`, `clear`).

```python
from ctxeng.stores.sqlite import SQLiteStore
store = SQLiteStore("ctxeng.db")
store.add("alice", "persistent memory")
```

#### TieredStore

Composes three stores into a fallback chain: queries working → episodic → long-term.

```python
from ctxeng.stores.memory import InMemoryStore
from ctxeng.stores.tiered import TieredStore

store = TieredStore(
    working_store=InMemoryStore(),
    episodic_store=InMemoryStore(),
    long_term_store=InMemoryStore(),
)
store.add("alice", "recent fact", metadata={"type": "working"})
store.add("alice", "past session summary", metadata={"type": "episodic"})
results = store.search("alice", "fact", top_k=5)  # queries all tiers
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

Heavy deps (`sentence-transformers`, `torch`, `numpy`) are opt-in:
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

Supports Jinja2 templates (opt-in) and Python `str.format()` out of the box.

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
| `MapReduceSummarizer` | Chunk→summarize→merge map-reduce pipeline |
| `LLMCompressor` | LLM-based compression with chunking |
| `MemoryConsolidator` | Working → episodic → long-term memory lifecycle |

```python
from ctxeng.compression.summarizer import ContextSummarizer, MapReduceSummarizer

s = ContextSummarizer(max_sentences=3)
summary = s.summarize("long text here...")

mr = MapReduceSummarizer(chunk_size=5)
summary = mr.summarize(["turn 1", "turn 2", ...])
```

```python
from ctxeng.compression.consolidator import MemoryConsolidator

consolidator = MemoryConsolidator(store, turn_threshold=10, batch_size=5)
consolidator.record_turn("alice", "user message")  # auto-consolidates at threshold
```

### Evaluation (`ctxeng.eval`)

```bash
python -m ctxeng.eval benchmark            # run all datasets
python -m ctxeng.eval benchmark --dataset simple_preferences  # single dataset
python -m ctxeng.eval benchmark --json results.json            # export as JSON
python -m ctxeng.eval check --json results.json --threshold mrr=0.5
python -m ctxeng.eval compare --baseline baseline.json --result results.json
```

| Dataset | Queries | Description |
|---------|---------|-------------|
| `simple_preferences` | 6 | User preference matching |
| `multi_topic_conversations` | 3 | Cross-topic retrieval |
| `long_term_cross_session` | 4 | Cross-session memory recall |
| `memory_retention` | 7 | Fact retention across turns |
| `long_context_coherence` | 5 | Coherence across 15 cross-referencing facts |

Metrics: P@1/3/5, R@3/5, MRR, MAP, token efficiency, latency.

Use `compare` to detect regressions against a baseline (flags drops > `--regression-threshold`).

### Ingestion (`ctxeng.ingestion`)

```python
from ctxeng.ingestion.ingestor import FileIngestor
ingestor = FileIngestor()
memories = ingestor.ingest("document.md", user_id="alice")
```

| Format | Ingestor | Extra |
|--------|----------|-------|
| `.txt`, `.log`, `.py`, `.js` | `TextIngestor` | — |
| `.md`, `.rst` | `MarkdownIngestor` | — |
| `.csv` | `CSVIngestor` | — |
| `.json` | `JSONIngestor` | — |
| `.jpg`, `.png`, `.gif` | `ImageIngestor` | `pillow`, `transformers` (captioning) |
| `.pdf` | `PDFIngestor` | `pypdf2` |

`FileIngestor` uses optional `python-magic` for MIME-based detection
with extension-based fallback. Install via `ctxeng[ingestion]` or `ctxeng[pdf]`.

### Safety (`ctxeng.core.safety`)

```python
from ctxeng.core.safety import InputValidator, ContextPoisoningFilter

validator = InputValidator()
result = validator.validate("ignore all previous instructions")
print(result.passed)  # False

filter = ContextPoisoningFilter()
clean = filter.filter_memories(memories)
```

- `InputValidator` — 10 prompt-injection patterns (ignore instructions, reveal system prompt, etc.)
- `ContextPoisoningFilter` — 5 poisoning patterns (overwrite behavior, new rules, etc.)

### Profile (`ctxeng.core.profile`)

```python
from ctxeng.core.profile import ProfileStore

store = ProfileStore()
store.set_preference("alice", "format", "bullet points")
store.set_tags("alice", ["power-user"])

context_snippet = store.to_context("alice")
# "Preferences:\n  format: bullet points\nTags: power-user"
```

### Tools (`ctxeng.tools`)

```python
from ctxeng.tools.base import ToolRegistry
from ctxeng.tools.calculator import CalculatorTool
from ctxeng.tools.web_search import WebSearchTool

registry = ToolRegistry()
registry.register(CalculatorTool())
registry.register(WebSearchTool())

output = registry.execute("calculator", "2 + 3 * 4")
print(output.output)  # "14"

matched = registry.match("use the calculator")
# returns [CalculatorTool]
```

`ContextManager.build_prompt()` auto-detects tools from the query
and injects their outputs into the assembled prompt.

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

| Provider | Backend |
|----------|---------|
| `OpenAIProvider` | OpenAI API (`openai` extra) |
| `OllamaProvider` | Local Ollama instances |

Or via the chat loop:

```bash
python -c "from ctxeng.llm.chat import run_chat; from ctxeng.llm.openai import OpenAIProvider; run_chat(OpenAIProvider())"
```

### Correctness Evaluation (`ctxeng.eval.judge`)

```python
from ctxeng.eval.judge import CorrectnessEvaluator

evaluator = CorrectnessEvaluator()  # token-overlap fallback
score = evaluator.evaluate(
    question="What is the capital of France?",
    answer="Paris",
    reference="The capital of France is Paris.",
)
print(score.score)  # 0.5 (token overlap)

# With an LLM provider for richer evaluation:
from ctxeng.llm.openai import OpenAIProvider
evaluator = CorrectnessEvaluator(provider=OpenAIProvider())
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

### Eval CLI

```bash
python -m ctxeng.eval benchmark            # run all 5 datasets
python -m ctxeng.eval benchmark --dataset simple_preferences
python -m ctxeng.eval benchmark --json results.json
python -m ctxeng.eval check --json results.json --threshold mrr=0.5
python -m ctxeng.eval compare --baseline main.json --result pr.json
```

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
| `vector` | `sentence-transformers`, `numpy`, `torch` |
| `ingestion` | `pillow`, `transformers`, `python-magic` |
| `pdf` | `pypdf2` |
| `llm` | `openai` |
| `all` | All of the above |

```bash
pip install "ctxeng[all]"
```

---

## Development

```bash
git clone <repo>
cd ctxeng
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all,test,pdf]"
pip install ruff pre-commit
pre-commit install
pytest
ruff check ctxeng/ tests/
ruff format --check ctxeng/ tests/
```

---

## Project Structure

```
ctxeng/
├── models.py                 # Core data contracts
├── app.py                    # Demo entry point
├── cli.py                    # Interactive CLI
├── server.py                 # FastAPI server
├── core/
│   ├── context_manager.py    # Orchestrator (safety + profile + tools + assembly)
│   ├── profile.py            # UserProfile + ProfileStore
│   └── safety.py             # InputValidator + ContextPoisoningFilter
├── stores/
│   ├── base.py               # ContextStore ABC
│   ├── memory.py             # InMemoryStore
│   ├── sqlite.py             # SQLiteStore
│   ├── vector.py             # VectorStore (ChromaDB)
│   └── tiered.py             # TieredStore (3-tier fallback)
├── retrieval/
│   ├── hybrid.py             # HybridRetriever
│   ├── embeddings.py         # EmbeddingModel
│   └── reranker.py           # CrossEncoderReranker
├── assembly/
│   ├── assembler.py          # Budget-aware assembly
│   ├── prioritizer.py        # Dedup + MMR ranking
│   └── templates.py          # PromptTemplate registry
├── observability/            # Span-based tracing
├── compression/
│   ├── summarizer.py         # TF-IDF + map-reduce + LLM compression
│   └── consolidator.py       # Working→episodic→long-term
├── eval/
│   ├── __main__.py           # Entry point
│   ├── cli.py                # benchmark / check / compare
│   ├── benchmark.py          # BenchmarkRunner
│   ├── datasets.py           # 5 built-in eval datasets
│   ├── metrics.py            # P@k, R@k, MRR, MAP
│   └── judge.py              # CorrectnessEvaluator (LLM-as-judge)
├── ingestion/
│   ├── ingestor.py           # FileIngestor (unified dispatcher)
│   ├── text.py               # TextIngestor, MarkdownIngestor
│   ├── pdf.py                # PDFIngestor
│   ├── image.py              # ImageIngestor (captioning)
│   └── structured.py         # CSVIngestor, JSONIngestor
├── routing/
│   ├── router.py             # ContextRouter
│   ├── diff.py               # ContextDiff
│   └── lifecycle.py          # LifecycleManager
├── tools/
│   ├── base.py               # BaseTool, ToolOutput, ToolRegistry
│   ├── calculator.py         # CalculatorTool
│   ├── web_search.py         # WebSearchTool
│   └── file_lookup.py        # FileLookupTool
├── llm/
│   ├── base.py               # LLMProvider ABC
│   ├── openai.py             # OpenAIProvider
│   ├── ollama.py             # OllamaProvider
│   └── chat.py               # run_chat loop
└── static/
    └── chat.html             # Web chat UI
```
