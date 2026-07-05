"""Retrieval evaluation harness.

Measures whether the retrieval step returns the RIGHT chunk for a set of
known questions. This is how you move from "I think it works" to a number
you can defend: "retrieval recall@k was X% on my eval set."

How it works:
  - Each test case is a question plus a keyword that MUST appear in a correctly
    retrieved chunk (a cheap, robust proxy for "the right passage").
  - We run the real retrieval pipeline (same embed + vector search the app uses).
  - A case "hits" if any of the top-k retrieved chunks contains the keyword.
  - We report the hit rate (recall@k) and show any misses for inspection.

Run (from backend/, venv active):  python -m scripts.eval_retrieval
Assumes the Acme handbook has been ingested.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db import get_conn, search_similar_chunks
from app.provider import embed_query
from app import config

# (question, keyword that should appear in the correct chunk)
TEST_CASES = [
    # From the JavaScript notes
    ("What order do the microtask and callback queues run in?", "Microtask"),
    ("What is a closure and why does a variable survive?", "reference"),
    ("What is a lexical environment compared to?", "backpack"),
    ("Difference between shallow copy and deep copy?", "recursively"),
    ("What method creates a deep copy in modern JavaScript?", "structuredClone"),
    ("How does JavaScript find toString on an empty object?", "prototype"),
    ("What does the prototype chain end with?", "null"),
    ("Why do prototypes save memory?", "share"),
    # From the resume
    ("What models were used in the building damage project?", "MaxViT"),
    ("What is the email on the resume?", "amulyabolusani123"),
    ("Which authentication did the Edify project use?", "JWT"),
]


def run_eval():
    conn = get_conn()
    hits = 0
    misses = []
    try:
        for question, keyword in TEST_CASES:
            q_emb = embed_query(question)
            rows = search_similar_chunks(conn, q_emb, config.TOP_K)
            # rows are (content, filename, similarity)
            retrieved_text = " ".join(content for (content, _f, _s) in rows).lower()
            top_sim = rows[0][2] if rows else 0.0
            if keyword.lower() in retrieved_text:
                hits += 1
                print(f"  HIT   (top sim {top_sim:.3f})  {question}")
            else:
                misses.append((question, keyword))
                print(f"  MISS  (top sim {top_sim:.3f})  {question}  [expected '{keyword}']")
    finally:
        conn.close()

    total = len(TEST_CASES)
    print("\n" + "=" * 50)
    print(f"Recall@{config.TOP_K}: {hits}/{total} = {hits / total:.0%}")
    if misses:
        print("\nMisses to investigate:")
        for q, kw in misses:
            print(f"  - {q!r} expected keyword {kw!r}")


if __name__ == "__main__":
    run_eval()
