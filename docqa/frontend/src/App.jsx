import { useState } from "react";
import AgentChat from "./AgentChat.jsx";

const API = "http://localhost:8000";

export default function App() {
  const [uploadStatus, setUploadStatus] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [asking, setAsking] = useState(false);
  const [error, setError] = useState(null);

  async function handleUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setUploadStatus(null);
    setError(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`${API}/upload`, { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Upload failed");
      setUploadStatus(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
      e.target.value = ""; // allow re-uploading the same file
    }
  }

  async function handleAsk() {
    if (!question.trim()) return;
    setAsking(true);
    setResult(null);
    setError(null);
    try {
      const res = await fetch(`${API}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Request failed");
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setAsking(false);
    }
  }

  return (
    <div className="page">
      <header>
        <h1>Doc Q&amp;A</h1>
        <p className="sub">Ask questions about your documents. Answers are grounded in retrieved passages and cite their sources.</p>
      </header>

      <section className="card">
        <h2>1 · Add a document</h2>
        <label className="upload">
          <input type="file" accept=".pdf,.txt,.md" onChange={handleUpload} disabled={uploading} />
          <span>{uploading ? "Ingesting…" : "Choose a PDF or text file"}</span>
        </label>
        {uploadStatus && (
          <p className="status">
            {uploadStatus.status === "ingested"
              ? `Ingested — ${uploadStatus.chunks} chunks stored.`
              : uploadStatus.status === "skipped"
              ? "Already ingested (identical file)."
              : JSON.stringify(uploadStatus)}
          </p>
        )}
      </section>

      <section className="card">
        <h2>2 · Ask a question</h2>
        <div className="ask-row">
          <input
            type="text"
            value={question}
            placeholder="What does the document say about…?"
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleAsk()}
          />
          <button onClick={handleAsk} disabled={asking}>
            {asking ? "Thinking…" : "Ask"}
          </button>
        </div>
      </section>

      {error && <p className="error">{error}</p>}

      {result && (
        <section className="card answer">
          <h2>Answer</h2>
          <p className="answer-text">{result.answer}</p>
          {result.sources?.length > 0 && (
            <details>
              <summary>Retrieved sources ({result.sources.length})</summary>
              <ul className="sources">
                {result.sources.map((s, i) => (
                  <li key={i}>
                    <div className="source-head">
                      <span>chunk {i + 1}</span>
                      <span className="sim">similarity {s.similarity}</span>
                    </div>
                    <p>{s.content}</p>
                  </li>
                ))}
              </ul>
            </details>
          )}
        </section>
      )}

      <section className="card">
        <AgentChat />
      </section>
    </div>
  );
}
