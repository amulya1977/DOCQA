# Doc Q&A — a RAG system with an agentic layer

Ask questions about your own documents and get answers **grounded in retrieved
passages with source citations** — plus an **AI agent** that decides, per
question, whether to search your documents, search the live web, or answer
directly, and that **self-corrects** when its first retrieval is weak.

The project has two layers:

1. **RAG core** — upload a document; it's split into chunks, embedded into
   vectors, and stored in Postgres (pgvector). A question is embedded the same
   way, the most similar chunks are retrieved by cosine similarity, and an LLM
   answers using only those chunks.
2. **Agent layer** — an LLM agent, built from scratch (no framework), that runs
   a reason -> act -> observe loop: it chooses tools (document search, web
   search, or direct answer), grades whether retrieved documents are relevant,
   rewrites the query and retries if not, and remembers the conversation for
   follow-ups.

The "AI" is just model calls; the engineering is everything around them —
chunking, idempotent ingestion, vector indexing, grounding to prevent
hallucination, tool-calling, and a self-correcting retrieval loop.

## Architecture

### RAG core

```
                    INGESTION (once per document)
  file -> extract text -> chunk -> embed (batch) -> store in pgvector
                                                        |
                    QUERY (every question)              |
  question -> embed -> similarity search (top-k) <------+
                              |
                              v
                  LLM answers from retrieved chunks, with citations
```

### Agent layer (reason -> act -> observe)

```
  question -> [ REASON: LLM decides what to do next ] <-----------+
                    |                                             |
        +-----------+-----------+                                 |
        v           v           v                                 |
  search_documents  web_search  answer directly                  |
  (self-grading:    (Tavily      |                                |
   grade -> rewrite  live web)   |                                |
   -> retry if weak)             |                                |
        +-----------+            |                                |
                    v            |                                |
             OBSERVE result -- not good? ------------------------+
                    | good       |                    (retry / rethink)
                    v            v
              [ Generate grounded, cited answer ]
```

- **Backend:** FastAPI (Python)
- **Vector store:** Postgres + pgvector (HNSW index, cosine distance)
- **Frontend:** React (Vite) — DocQA panel + agent chat with a live tool-trace
- **LLM provider:** OpenAI-compatible, isolated behind `app/provider.py`
- **Web search:** Tavily (search API built for agents), isolated in `app/tools.py`

## Project layout

```
backend/
  app/
    config.py      # all tunable knobs in one place (models, chunk size, keys)
    provider.py    # the ONLY file that talks to the embedding/LLM provider
    db.py          # all SQL lives here
    chunking.py    # split text into overlapping windows
    extract.py     # PDF / text -> plain text
    ingest.py      # text -> chunks -> embeddings -> DB (idempotent)
    query.py       # plain RAG: question -> embed -> retrieve -> generate
    tools.py       # agent tools: document search (self-grading) + web search
    agent.py       # the hand-written reason->act->observe agent loop
    main.py        # FastAPI endpoints: /upload, /ask, /agent, /health
  scripts/
    ingest_cli.py     # ingest files from the command line
    eval_retrieval.py # measures retrieval quality (recall@k) on a test set
  schema.sql       # tables + pgvector extension + index
frontend/          # React UI (DocQA panel + agent chat)
sample_docs/       # a handbook to test with
```

## Setup

### 1. Database (Postgres + pgvector)

Either run it with Docker:

```bash
docker compose up -d   # applies schema.sql automatically
```

...or use a local Postgres install with the `pgvector` extension, then apply the
schema manually:

```bash
psql -U postgres -c "CREATE DATABASE docqa;"
psql -U postgres -d docqa -c "CREATE EXTENSION vector;"
psql -U postgres -d docqa -f backend/schema.sql
```

### 2. Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`:

```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...          # optional; enables live web search
DATABASE_URL=postgresql://postgres:PASSWORD@localhost:5432/docqa
```

Run it, and ingest the sample doc to verify:

```bash
uvicorn app.main:app --reload --port 8000
python -m scripts.ingest_cli ../sample_docs/acme_handbook.txt
```

### 3. Frontend

```bash
cd frontend
npm install
npm run dev      # http://localhost:5173
```

Upload a document and try both panels: the **DocQA** panel (plain RAG with
sources), and the **Agent** panel — ask a document question, a live-web question
("latest AI news"), or a general one, and expand "what the agent did" to see
which tools it chose.

## Design decisions (and the tradeoffs)

**Agent built by hand, not with a framework.** The reason->act->observe loop and
tool-calling are implemented directly in `agent.py`, so every decision is visible
and explainable rather than hidden behind a framework abstraction. A framework
(LangGraph) would be the next step for more complex multi-step graphs.

**Tool-calling for routing.** The LLM is given tool schemas and decides which to
call (document search, web search) or whether to answer directly. Good tool
*descriptions* are effectively prompt engineering — they drive good routing.

**Self-grading retrieval.** After a document search, a cheap LLM call grades
whether the chunks are actually relevant; if not, the query is rewritten and
retrieval retried once. This trades a small extra cost for higher reliability —
plain RAG cannot recover from a bad first search.

**Web search via Tavily.** A search API designed for agents (clean, ranked,
LLM-ready snippets). Isolated in `tools.py` so swapping providers touches one
function.

**Chunking — fixed-size with overlap.** Simple and explainable. Larger chunks
carry more context but retrieve less precisely; smaller chunks retrieve sharply
but lose context. Overlap prevents a sentence at a boundary from being cut in
half. Semantic chunking is a natural next step.

**pgvector over a dedicated vector DB.** One datastore, ACID guarantees, no extra
infra. A purpose-built store (Pinecone, Weaviate) is worth it only at much larger
scale — choosing the boring option first is the point.

**HNSW index, cosine distance.** Cosine compares direction, not magnitude — the
right choice for text embeddings (meaning, not length). HNSW gives fast
approximate nearest-neighbour search that scales past a linear scan.

**Idempotent ingestion via content hash.** Each document is fingerprinted with
SHA-256 of its text, so re-uploading the same file is a no-op instead of creating
duplicate vectors.

**Grounding to prevent hallucination.** Generation is instructed to answer only
from retrieved context, cite sources, and say when the answer isn't present.

**Provider isolation.** Everything provider-specific lives in `provider.py`
(embeddings/LLM) and `tools.py` (web search), so swapping models or search
providers is a one-file change.

## Evaluation

`scripts/eval_retrieval.py` measures retrieval quality: it runs a set of
question -> expected-keyword pairs through the real retrieval pipeline and
reports recall@k, surfacing which questions retrieval misses so chunking and
parameters can be tuned against a number instead of a guess.

## Known limitations / next steps

- Ingestion is synchronous; large documents would benefit from a background
  queue so uploads don't block.
- No re-ranking step after retrieval (a cross-encoder re-ranker would sharpen
  precision at scale).
- Single-user, no auth or multi-tenancy (chunks are tagged by document, not user).
- Chunking is character-based, not semantic.
- Web search requires a Tavily API key; without it, the agent falls back to
  document search and direct answers.
