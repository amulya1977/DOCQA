"""Query orchestrator: question -> embed -> retrieve -> generate.

The whole of Phase 2, and again mostly glue. The retrieval happens in db.py,
the generation in provider.py; this file just sequences them.
"""
from . import db, provider, config


def answer_question(conn, question: str) -> dict:
    # 1. Embed the question with the SAME model used for chunks.
    q_emb = provider.embed_query(question)

    # 2. Retrieve the top-k most similar chunks.
    rows = db.search_similar_chunks(conn, q_emb, config.TOP_K)
    if not rows:
        return {"answer": "No documents have been ingested yet.", "sources": []}

    contexts = [content for (content, _fname, _sim) in rows]
    sources = [
        {"content": content, "source": filename, "similarity": round(float(sim), 3)}
        for (content, filename, sim) in rows
    ]

    # 3. Generate a grounded, cited answer from those chunks.
    answer = provider.generate_answer(question, contexts)

    return {"answer": answer, "sources": sources}
