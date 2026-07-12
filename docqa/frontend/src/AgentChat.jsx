import { useState } from "react";

const API = "http://localhost:8000";

// A chat interface for the agent. Shows the conversation, and for each answer
// an expandable trace of what the agent DID (which tools it called) — so you
// can see the reason->act->observe loop in action, not just the final answer.
export default function AgentChat() {
  const [messages, setMessages] = useState([]); // {role, content, steps?}
  const [input, setInput] = useState("");
  const [thinking, setThinking] = useState(false);
  const [error, setError] = useState(null);

  async function send() {
    const question = input.trim();
    if (!question || thinking) return;
    setError(null);
    setInput("");

    // Show the user's message immediately.
    const newMessages = [...messages, { role: "user", content: question }];
    setMessages(newMessages);
    setThinking(true);

    try {
      // Send prior turns as history so the agent has follow-up memory.
      // We only send role+content (not the step traces).
      const history = messages.map((m) => ({ role: m.role, content: m.content }));

      const res = await fetch(`${API}/agent`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, history }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Agent request failed");

      setMessages([
        ...newMessages,
        { role: "assistant", content: data.answer, steps: data.steps },
      ]);
    } catch (err) {
      setError(err.message);
    } finally {
      setThinking(false);
    }
  }

  return (
    <div className="agent">
      <h2>Agent</h2>
      <p className="agent-sub">
        Ask anything. The agent decides whether to search your documents, search
        the web, or answer directly — and remembers the conversation.
      </p>

      <div className="chat">
        {messages.length === 0 && (
          <p className="empty">No messages yet. Ask a question to start.</p>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            <div className="bubble">{m.content}</div>
            {m.steps && m.steps.length > 0 && <AgentTrace steps={m.steps} />}
          </div>
        ))}
        {thinking && <div className="msg assistant"><div className="bubble thinking">Thinking…</div></div>}
      </div>

      {error && <p className="error">{error}</p>}

      <div className="chat-input">
        <input
          type="text"
          value={input}
          placeholder="Ask the agent…"
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
          disabled={thinking}
        />
        <button onClick={send} disabled={thinking}>Send</button>
      </div>
    </div>
  );
}

// Shows the agent's internal steps for one answer: which tools it called and
// a preview of what came back. This is the "reason -> act -> observe" trace.
function AgentTrace({ steps }) {
  const [open, setOpen] = useState(false);
  const toolSteps = steps.filter((s) => s.type === "tool_call" || s.type === "observation");
  if (toolSteps.length === 0) {
    return <div className="trace-note">answered directly (no tools used)</div>;
  }
  return (
    <details className="trace" open={open} onToggle={(e) => setOpen(e.target.open)}>
      <summary>what the agent did ({toolSteps.filter(s => s.type === "tool_call").length} tool call(s))</summary>
      <ul>
        {steps.map((s, i) => {
          if (s.type === "tool_call")
            return <li key={i}><b>→ called</b> {s.tool}({JSON.stringify(s.args)})</li>;
          if (s.type === "observation")
            return <li key={i} className="obs"><b>← got</b> {s.result_preview}…</li>;
          return null;
        })}
      </ul>
    </details>
  );
}
