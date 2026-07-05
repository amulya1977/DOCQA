# Doc Q&A — a small, honest RAG system

Ask questions about your own documents and get answers that are **grounded in
retrieved passages and cite their sources**. Upload a PDF or text file; it gets
split into chunks, embedded into vectors, and stored in Postgres (pgvector). A
question is embedded the same way, the most similar chunks are retrieved by
cosine similarity, and an LLM answers using only those chunks.

This project is deliberately small. The "AI" is two model calls (embed +
generate); the engineering is everything around them — chunking, batching,
idempotent ingestion, vector indexing, and grounding to prevent hallucination.

## Architecture

```
                    INGESTION (once per document)
  file ──▶ extract text ──▶ chunk ──▶ embed (batch) ──▶ store in pgvector
                                                            │
                    QUERY (every question)                  │
  question ──▶ embed ──▶ similarity search (top-k) ◀────────┘
                              │
                              ▼
                  LLM answers from retrieved chunks, with citations
```

- **Backend:** FastAPI (Python)
- **Vector store:** Postgres + pgvector (HNSW index, cosine distance)
- **Frontend:** React (Vite)
- **Provider:** OpenAI-compatible, isolated behind `app/provider.py`

## Project layout

```
backend/
  app/
    config.py      # all tunable knobs in one place
    provider.py    # the ONLY file that talks to the AI provider
    db.py          # all SQL lives here
    chunking.py    # split text into overlapping windows
    extract.py     # PDF / text -> plain text
    ingest.py      # text -> chunks -> embeddings -> DB (idempotent)
    query.py       # question -> embed -> retrieve -> generate
    main.py        # FastAPI endpoints: /upload, /ask, /health
  scripts/
    ingest_cli.py  # ingest files from the command line
  schema.sql       # tables + pgvector extension + index
frontend/          # React single-page UI
docker-compose.yml # Postgres + pgvector
sample_docs/       # a handbook to test with
```

## Setup

### 1. Start the database

```bash
docker compose up -d
```

This runs Postgres with pgvector and applies `schema.sql` automatically.

### 2. Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then put your OPENAI_API_KEY in .env
uvicorn app.main:app --reload --port 8000
```

Ingest the sample doc to verify everything works:

```bash
python -m scripts.ingest_cli ../sample_docs/acme_handbook.txt
```

### 3. Frontend

```bash
cd frontend
npm install
npm run dev      # opens http://localhost:5173
```

Upload a document, ask a question (e.g. "How long is the free trial?"), and the
answer appears with its retrieved source chunks and similarity scores.

## Design decisions (and the tradeoffs)

**Chunking — fixed-size with overlap.** Simple and explainable. Larger chunks
carry more context but retrieve less precisely; smaller chunks retrieve sharply
but lose context. Overlap (150 chars) prevents a sentence that straddles a
boundary from being cut in half. A natural next step is semantic chunking
(splitting on paragraph/sentence boundaries).

**pgvector over a dedicated vector DB.** Keeps the stack to one datastore I
already run, with ACID guarantees and no extra infra. At millions of vectors or
with heavy filtering needs, a purpose-built store (Pinecone, Weaviate) becomes
worth the operational cost — but choosing the boring option first is the point.

**HNSW index, cosine distance.** Cosine compares direction, not magnitude, which
is what we want for text embeddings (meaning, not length). HNSW gives fast
approximate nearest-neighbour search that scales past a linear scan.

**Idempotent ingestion via content hash.** Each document is fingerprinted with
SHA-256 of its text. Re-uploading the same file is a no-op instead of creating
duplicate vectors that would skew retrieval.

**Batched embeddings.** All chunks of a document are embedded in one API call —
far cheaper and faster than one call per chunk.

**Grounding to prevent hallucination.** The generation prompt instructs the model
to answer only from the retrieved chunks, to cite chunk numbers, and to say when
the answer isn't present. Temperature 0 keeps it faithful rather than creative.

**Provider isolation.** Everything provider-specific lives in `provider.py`.
Swapping to a local `sentence-transformers` embedding model, or to a different
LLM for generation, touches only that one file.

## Known limitations / next steps

- No async ingestion queue — large files block the request. A job queue would fix this.
- No re-ranking step after retrieval (a cross-encoder re-ranker would improve precision).
- No auth or multi-tenancy.
- Chunking is character-based, not semantic.
