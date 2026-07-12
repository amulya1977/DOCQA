"""The agent loop, implemented by hand (no framework).

This is the whole idea of an "agent": an LLM in a loop that decides what to do,
does it, looks at the result, and decides again — until it produces a final
answer. The three steps repeat:

    REASON  -> the LLM looks at the conversation and decides: call a tool, or answer?
    ACT     -> if it chose a tool, we run that tool
    OBSERVE -> we feed the tool's result back into the conversation
    (loop)  -> back to REASON with the new information

The LLM drives every decision. We (the code) just execute the tools it requests
and feed results back. That inversion — LLM decides, code obeys — is what makes
this an agent instead of a fixed pipeline.

How the LLM "chooses a tool": we pass TOOL_SCHEMAS to the chat model. Modern
models support "tool calling" — instead of returning text, the model can return
a structured request: "call search_documents with query='...'." We detect that,
run the tool, append the result as a 'tool' message, and call the model again.
When the model finally returns plain text (no tool call), that's the answer.
"""
from openai import OpenAI
from . import config
from .tools import TOOL_SCHEMAS, run_tool

_client = OpenAI(api_key=config.OPENAI_API_KEY)

# Safety cap: never loop forever. If the agent hasn't answered in this many
# steps, we stop. (Runaway loops are the #1 agent failure mode.)
MAX_STEPS = 6

SYSTEM_PROMPT = (
    "You are a helpful research assistant with access to tools.\n"
    "- Use search_documents when the question is about the user's own documents.\n"
    "- Use web_search when the answer needs current or general web information.\n"
    "- If you can answer directly without any tool, just answer.\n"
    "After using a tool, look at the result: if it doesn't answer the question, "
    "you may search again with a better query. When a document tool returns "
    "relevant chunks, ground your answer in them and cite the source. "
    "If you cannot find the answer, say so honestly."
)


def run_agent(question: str, history: list | None = None) -> dict:
    """Run the agent loop for one user question.

    `history` (optional) is prior conversation turns for follow-up memory:
    a list of {"role": "user"/"assistant", "content": "..."} messages.

    Returns {"answer": str, "steps": [...]} where steps is a trace of what the
    agent did (useful for the UI and for understanding/debugging).
    """
    # Build the running conversation the model sees. It starts with the system
    # instructions, any prior history, then the new question.
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": question})

    trace = []  # human-readable record of each step, for transparency

    for step in range(MAX_STEPS):
        # ---- REASON: ask the model what to do next ----
        # We pass the tools so the model *can* choose to call one.
        response = _client.chat.completions.create(
            model=config.GENERATION_MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",   # let the model decide: tool or plain answer
            temperature=0.0,
        )
        msg = response.choices[0].message

        # Did the model ask to call one or more tools?
        tool_calls = msg.tool_calls

        if not tool_calls:
            # ---- No tool requested => this is the final answer. Done. ----
            trace.append({"type": "answer", "content": msg.content})
            return {"answer": msg.content or "", "steps": trace}

        # The model wants to use tools. We must add its request to the
        # conversation first (the API requires the assistant tool-call message
        # to precede the tool results).
        messages.append(
            {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
        )

        # ---- ACT + OBSERVE: run each requested tool, feed results back ----
        import json
        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            trace.append({"type": "tool_call", "tool": name, "args": args})

            result = run_tool(name, args, question=question)   # ACT: run the real function
            trace.append({"type": "observation", "tool": name,
                          "result_preview": result[:300]})

            # OBSERVE: give the result back to the model as a 'tool' message,
            # linked to the call by id. Next loop iteration, the model sees it.
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
            )
        # loop back to REASON, now with the tool results in context

    # Ran out of steps without a final answer.
    return {
        "answer": "I wasn't able to reach a confident answer within the step limit.",
        "steps": trace,
    }
