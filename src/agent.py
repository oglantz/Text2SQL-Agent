"""Model-agnostic ReAct agent loop. FROZEN after Phase 0 — never change its behavior.

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

SYSTEM_PROMPT = """\
You are a SQL agent. Explore the database with the provided tools, then write a
correct SQLite query and finish by calling submit(sql). TODO: flesh out — describe
each tool and require the model to end with submit.
"""


def dispatch_tool(name, db_id, args):
    raise NotImplementedError


def run_episode(client, model, db_id, question, tool_defs, max_steps=8):
    raise NotImplementedError
