"""FastAPI app — the HTTP surface of the project.

Three endpoints:
  GET  /health        -> liveness check
  POST /upload        -> ingest a document (PDF or text)
  POST /ask           -> ask a question, get a grounded cited answer

Deliberately small. The intelligence lives in the service modules; this file
just wires HTTP to them and handles errors.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .db import get_conn
from .extract import extract_text
from .ingest import ingest_text
from .query import answer_question
from .agent import run_agent

app = FastAPI(title="Doc Q&A (RAG)")

# Allow the React dev server to call us during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


class AgentMessage(BaseModel):
    role: str      # "user" or "assistant"
    content: str


class AgentRequest(BaseModel):
    question: str
    history: list[AgentMessage] = []   # prior turns, for follow-up memory


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    text = extract_text(file.filename, raw)
    if not text.strip():
        raise HTTPException(status_code=400, detail="No text could be extracted from the file.")

    conn = get_conn()
    try:
        result = ingest_text(conn, file.filename, text)
    finally:
        conn.close()

    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result["reason"])
    return result


@app.post("/ask")
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    conn = get_conn()
    try:
        return answer_question(conn, question)
    finally:
        conn.close()


@app.post("/agent")
def agent(req: AgentRequest):
    """Agentic endpoint: the LLM decides which tools to use (document search,
    web search, or answer directly), can self-correct, and supports follow-up
    memory via the history field."""
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    history = [{"role": m.role, "content": m.content} for m in req.history]
    return run_agent(question, history)
