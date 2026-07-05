"""The one abstraction that signals good design instinct in an interview.

Everything that talks to an AI provider goes through these two functions.
Swapping OpenAI for a local sentence-transformers model, or for Anthropic's
API for generation, means editing only this file. Nothing else in the codebase
knows or cares which provider is behind these calls.
"""
from openai import OpenAI
from . import config

_client = OpenAI(api_key=config.OPENAI_API_KEY)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Turn a batch of strings into embedding vectors.

    Batched on purpose: one API call for many chunks is far cheaper and faster
    than one call per chunk. This is the function you'd swap to go local.
    """
    if not texts:
        return []
    # The embedding API caps how many inputs you can send per request, so for
    # large documents (thousands of chunks) we send in batches and stitch the
    # results back together in order.
    batch_size = 100
    all_embeddings = []
    total = len(texts)
    for start in range(0, total, batch_size):
        batch = texts[start:start + batch_size]
        resp = _client.embeddings.create(model=config.EMBEDDING_MODEL, input=batch)
        # The API returns results in the same order we sent them.
        all_embeddings.extend(item.embedding for item in resp.data)
        done = min(start + batch_size, total)
        print(f"    embedded {done}/{total} chunks")
    return all_embeddings


def embed_query(text: str) -> list[float]:
    """Embed a single question. MUST use the same model as the chunks,
    or the query vector won't live in the same space as the stored vectors."""
    return embed_texts([text])[0]


def generate_answer(question: str, context_chunks: list[str]) -> str:
    """Ask the LLM to answer ONLY from the retrieved context, with citations.

    This grounding is what prevents hallucination — the model is instructed to
    rely on the provided chunks and to say so when the answer isn't there.
    """
    context = "\n\n".join(
        f"[chunk {i + 1}]\n{c}" for i, c in enumerate(context_chunks)
    )
    system = (
        "You answer questions strictly using the provided context chunks. "
        "Cite the chunk number(s) you used, like [chunk 2]. "
        "If the answer is not in the context, say you don't find it in the documents. "
        "Do not use outside knowledge."
    )
    user = f"Context:\n{context}\n\nQuestion: {question}"
    resp = _client.chat.completions.create(
        model=config.GENERATION_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,  # deterministic: we want grounded recall, not creativity
    )
    return resp.choices[0].message.content or ""
