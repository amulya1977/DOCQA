"""Chunking: split a long text into overlapping windows.

This is deliberately simple (character-based, fixed size) so you can explain
every line. The overlap means a sentence that straddles a boundary still
appears whole in at least one chunk — preventing "cut in half" retrieval misses.

The tradeoff to articulate in an interview:
  - larger chunks  -> more context per chunk, but less precise retrieval
  - smaller chunks -> sharper retrieval, but each chunk may lack context
  - overlap        -> guards boundaries, at the cost of some duplication
A natural next step is semantic chunking (split on paragraphs/sentences),
which you can mention as a known improvement.
"""
from . import config


def chunk_text(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []

    size = config.CHUNK_SIZE
    overlap = config.CHUNK_OVERLAP
    step = size - overlap  # how far we advance each window

    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks
