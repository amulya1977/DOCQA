"""Thin database layer. All SQL lives here so the rest of the app deals in
Python, not query strings. Uses psycopg3 with the pgvector adapter so we can
pass Python lists straight into vector columns.
"""
import psycopg
from pgvector.psycopg import register_vector
from . import config


def get_conn():
    """Open a connection and teach psycopg about the vector type."""
    conn = psycopg.connect(config.DATABASE_URL)
    register_vector(conn)
    return conn


def find_document_by_hash(conn, content_hash: str):
    """Return the document row id if this exact content was already ingested."""
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM documents WHERE content_hash = %s", (content_hash,))
        row = cur.fetchone()
        return row[0] if row else None


def insert_document(conn, filename: str, content_hash: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO documents (filename, content_hash) VALUES (%s, %s) RETURNING id",
            (filename, content_hash),
        )
        doc_id = cur.fetchone()[0]
    conn.commit()
    return doc_id


def insert_chunks(conn, document_id: int, chunks: list[str], embeddings: list[list[float]]):
    """Bulk-insert chunk text alongside its embedding vector.

    Uses executemany with batching so a large document (thousands of chunks)
    is a handful of round-trips instead of one per row.
    """
    rows = []
    for idx, (content, emb) in enumerate(zip(chunks, embeddings)):
        emb_str = "[" + ",".join(str(x) for x in emb) + "]"
        rows.append((document_id, idx, content, emb_str))

    batch_size = 500
    with conn.cursor() as cur:
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            cur.executemany(
                "INSERT INTO chunks (document_id, chunk_index, content, embedding) "
                "VALUES (%s, %s, %s, %s::vector)",
                batch,
            )
    conn.commit()


def search_similar_chunks(conn, query_embedding: list[float], top_k: int):
    """The core retrieval query.

    `embedding <=> %s` is cosine DISTANCE (0 = identical direction, 2 = opposite).
    Ordering ascending gives the most similar chunks first. We return a
    similarity score (1 - distance) just for display / debugging.
    """
    with conn.cursor() as cur:
        emb_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
        cur.execute(
            "SELECT c.content, d.filename, 1 - (c.embedding <=> %s::vector) AS similarity "
            "FROM chunks c JOIN documents d ON c.document_id = d.id "
            "ORDER BY c.embedding <=> %s::vector LIMIT %s",
            (emb_str, emb_str, top_k),
        )
        return cur.fetchall()  # list of (content, filename, similarity)
