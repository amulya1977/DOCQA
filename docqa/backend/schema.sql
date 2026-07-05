-- Enable the pgvector extension. This adds the `vector` column type
-- and distance operators like <=> (cosine distance).
CREATE EXTENSION IF NOT EXISTS vector;

-- One row per ingested document. The content_hash makes ingestion
-- idempotent: re-uploading the same file is a no-op instead of a duplicate.
CREATE TABLE IF NOT EXISTS documents (
    id            BIGSERIAL PRIMARY KEY,
    filename      TEXT        NOT NULL,
    content_hash  TEXT        NOT NULL UNIQUE,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per chunk. embedding is a 1536-dim vector (OpenAI text-embedding-3-small).
-- If you switch embedding models, change the dimension to match.
CREATE TABLE IF NOT EXISTS chunks (
    id           BIGSERIAL PRIMARY KEY,
    document_id  BIGINT      NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index  INT         NOT NULL,
    content      TEXT        NOT NULL,
    embedding    vector(1536) NOT NULL
);

-- An approximate-nearest-neighbour index for fast cosine search at scale.
-- vector_cosine_ops tells the index we'll query with cosine distance (<=>).
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks(document_id);
