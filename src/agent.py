"""
Model-agnostic ReAct agent loop. FROZEN after Phase 0 — never change its behavior.

Talks to any OpenAI-compatible chat-completions endpoint with tool calling, so the
same code drives both teacher (hosted API) and student (vLLM). Swap base_url + model.

    run_episode(client, model, db_id, question, tool_defs, max_steps=8) -> dict
        Returns {"submitted_sql", "messages", "steps"}.
        The full `messages` list IS the trajectory — save it verbatim; it's the
        training data in Phase 1.

    dispatch_tool(name, db_id, args) -> str
        Routes a tool call to src/tools.py, returns a string-safe result,
        truncating result sets to the first 30 rows so context doesn't explode.

Placeholder — implement in Phase 0.
"""
import json
from openai import OpenAI
from src.tools import dispatch_tool

SYSTEM_PROMPT = """\
You are a SQL agent. Explore the database with the provided tools, then write a
correct SQLite query and finish by calling submit(sql). TODO: flesh out — describe
each tool and require the model to end with submit.
"""




def run_episode(client, model, db_id, question, tool_defs, max_steps=8):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},   # explains tools + that it must call submit()
        {"role": "user", "content": f"Database: {db_id}\nQuestion: {question}"},
    ]
    for step in range(max_steps):
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=tool_defs, temperature=0.0,
        )
        msg = resp.choices[0].message
        messages.append(msg)                              # record the assistant turn (thoughts + tool call)

        if not msg.tool_calls:                            # model answered without a tool — nudge or stop
            continue

        for call in msg.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments)
            if name == "submit":
                return {"submitted_sql": args["sql"], "messages": messages, "steps": step + 1}
            result = dispatch_tool(name, db_id, args)     # calls into src/tools.py
            messages.append({                             # feed the OBSERVATION back
                "role": "tool", "tool_call_id": call.id, "content": str(result),
            })
    return {"submitted_sql": None, "messages": messages, "steps": max_steps}  # gave up
