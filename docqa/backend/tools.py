"""The agent's tools.

A "tool" is just a function the LLM is allowed to request. Each tool here has:
  - a real Python function that does the work
  - a JSON schema describing it (name, what it does, its arguments)

The schema is how the LLM *learns what tools exist*. We hand these schemas to
the model; the model reads the descriptions and decides which tool to call and
with what arguments. Good descriptions = good tool choices, so the text in
"description" is effectively prompt engineering.

We deliberately keep the two retrieval-style tools separate:
  - search_documents: looks inside the user's ingested documents (your DocQA),
    with SELF-GRADING: it checks whether the retrieved chunks are actually
    relevant, and if not, rewrites the query and tries once more.
  - web_search: looks on the public internet for current/general info
The agent picks between them based on the question.
"""
import json
from openai import OpenAI
from . import db, provider, config

_client = OpenAI(api_key=config.OPENAI_API_KEY)


# ---- Self-grading helper: is this retrieval good enough? ----

def _grade_relevance(question: str, chunks_text: str) -> bool:
    """Ask the LLM: do these retrieved chunks actually help answer the question?

    Returns True if relevant, False if not. This is the 'grader' that makes the
    retrieval self-correcting — a cheap yes/no LLM call. If chunks are judged
    irrelevant, the caller rewrites the query and retries.
    """
    prompt = (
        "You are grading whether retrieved text is relevant to a question.\n"
        f"Question: {question}\n\n"
        f"Retrieved text:\n{chunks_text[:2000]}\n\n"
        "Does the retrieved text contain information useful for answering the "
        "question? Answer with exactly one word: yes or no."
    )
    try:
        resp = _client.chat.completions.create(
            model=config.GENERATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        verdict = (resp.choices[0].message.content or "").strip().lower()
        return verdict.startswith("yes")
    except Exception:
        # If grading fails, don't block — assume relevant and proceed.
        return True


def _rewrite_query(question: str, original_query: str) -> str:
    """Ask the LLM to produce a better search query when the first retrieval
    was judged irrelevant — e.g. different keywords, more specific terms."""
    prompt = (
        "A document search returned irrelevant results.\n"
        f"User question: {question}\n"
        f"Query that failed: {original_query}\n\n"
        "Write a better search query (different or more specific keywords) that "
        "is more likely to retrieve relevant passages. Return ONLY the new query."
    )
    try:
        resp = _client.chat.completions.create(
            model=config.GENERATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return (resp.choices[0].message.content or original_query).strip()
    except Exception:
        return original_query


# ---- Tool 1: search the user's documents (self-grading + rewrite) ----

def _raw_document_search(query: str) -> str:
    """One retrieval pass: embed the query, fetch top chunks, format as text."""
    conn = db.get_conn()
    try:
        q_emb = provider.embed_query(query)
        rows = db.search_similar_chunks(conn, q_emb, config.TOP_K)
    finally:
        conn.close()
    if not rows:
        return ""
    parts = []
    for i, (content, filename, sim) in enumerate(rows):
        parts.append(f"[chunk {i+1} | source: {filename} | similarity: {sim:.3f}]\n{content}")
    return "\n\n".join(parts)


def search_documents(query: str, question: str = "") -> str:
    """Search the user's documents, GRADING the result and RETRYING once with a
    rewritten query if the first attempt looks irrelevant.

    `question` is the original user question (used for grading/rewriting). If not
    provided, we fall back to the query itself.
    """
    q_for_grading = question or query

    # First attempt
    result = _raw_document_search(query)
    if not result:
        return "NO_RESULTS: no documents have been ingested, or nothing matched."

    # Grade it. If good, return. If not, rewrite the query and try once more.
    if _grade_relevance(q_for_grading, result):
        return result

    better_query = _rewrite_query(q_for_grading, query)
    if better_query and better_query != query:
        retry = _raw_document_search(better_query)
        if retry:
            # Prefix so the trace shows the self-correction happened.
            return f"(retried with query: {better_query})\n\n{retry}"

    # Return the original even if imperfect — better than nothing; the agent
    # can still decide it's insufficient and say so.
    return result


# ---- Tool 2: search the web (for current / general questions) ----

def web_search(query: str) -> str:
    """Search the public web with Tavily. Used when the answer isn't in the
    user's documents (current events, general knowledge, things outside the files).

    Tavily is a search API built for AI agents — it returns clean, ranked
    snippets ready to feed to an LLM. Free tier: 1000 searches/month.
    Isolated here so swapping search providers touches only this function.
    """
    if not config.TAVILY_API_KEY:
        return (
            "WEB_SEARCH_UNAVAILABLE: no TAVILY_API_KEY configured. "
            "Add one to .env to enable live web search."
        )
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=config.TAVILY_API_KEY)
        # basic depth keeps it cheap (1 credit); include a short synthesized answer
        resp = client.search(query, search_depth="basic", max_results=5, include_answer=True)
    except Exception as e:
        return f"WEB_SEARCH_ERROR: {e}"

    parts = []
    # Tavily can return a short direct answer plus ranked source snippets.
    if resp.get("answer"):
        parts.append(f"[web answer] {resp['answer']}")
    for i, r in enumerate(resp.get("results", [])[:5]):
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "")
        parts.append(f"[web result {i+1} | {title} | {url}]\n{content}")

    if not parts:
        return f"NO_WEB_RESULTS for '{query}'."
    return "\n\n".join(parts)


# ---- Tool registry: the schemas the LLM sees ----
# This is the list we hand to the model so it knows what it can call.
# The format matches OpenAI's tool-calling schema.

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search the user's uploaded/ingested documents for relevant passages. "
                "Use this when the question is about the content of the user's documents "
                "(their files, notes, resume, handbook, etc.)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A focused search query with the key terms from the question.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the public internet. Use this when the answer is NOT likely in "
                "the user's documents — for current events, recent news, or general "
                "knowledge outside their files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A focused web search query.",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


# Maps a tool name (as the LLM calls it) to the actual Python function.
TOOL_FUNCTIONS = {
    "search_documents": search_documents,
    "web_search": web_search,
}


def run_tool(name: str, arguments: dict, question: str = "") -> str:
    """Execute a tool by name with the given arguments, returning its text result.

    This is the 'Act' step: the LLM asked for a tool, we run the real function.
    `question` is the original user question, injected into search_documents so
    the self-grading step knows what "relevant" means.
    """
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return f"ERROR: unknown tool '{name}'"
    try:
        # Give the document search the original question for grading/rewriting.
        if name == "search_documents":
            arguments = {**arguments, "question": question}
        return func(**arguments)
    except Exception as e:
        # Return the error as an observation so the agent can react, not crash.
        return f"ERROR running {name}: {e}"
