"""Ingestion orchestrator: text -> chunks -> embeddings -> database.

This is the whole of Phase 1. Notice it's just glue: the interesting decisions
(chunking, batching, dedup) live in the functions it calls.
"""
import hashlib
from . import db, provider
from .chunking import chunk_text


def content_hash(text: str) -> str:
    """A fingerprint of the document's text. Identical text -> identical hash,
    which is how we detect and skip re-ingestion."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ingest_text(conn, filename: str, text: str) -> dict:
    h = content_hash(text)

    # Idempotency check: already ingested? Do nothing.
    existing = db.find_document_by_hash(conn, h)
    if existing is not None:
        return {"status": "skipped", "reason": "already ingested", "document_id": existing}

    chunks = chunk_text(text)
    if not chunks:
        return {"status": "error", "reason": "no text extracted"}

    print(f"  {filename}: {len(chunks)} chunks — embedding...")
    embeddings = provider.embed_texts(chunks)
    print(f"  {filename}: storing {len(chunks)} chunks...")
    doc_id = db.insert_document(conn, filename, h)
    db.insert_chunks(conn, doc_id, chunks, embeddings)

    return {"status": "ingested", "document_id": doc_id, "chunks": len(chunks)}
