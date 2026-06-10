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
You are an expert text-to-SQL agent. Your job: given a natural-language question
about a specific SQLite database, discover the database's structure and contents
with the tools provided, then produce ONE correct SQLite query that answers the
question, and finish by calling `submit`.

You do not know the schema or the data in advance. Do not guess table names,
column names, or stored values — every one of them is discoverable with a tool,
and guessing is the single biggest cause of wrong answers. Investigate first,
write SQL second.

# Tools

- list_tables()
    Lists every table name in the database. Cheap. Call this FIRST to orient
    yourself before assuming anything exists.

- get_schema(tables=None)
    Returns the CREATE TABLE statements — columns, types, primary keys, and
    foreign keys — for the named tables, or for all tables if you omit `tables`.
    This is your map: it tells you which columns live in which table and how
    tables join (follow the foreign keys). Read it before writing any SQL.

- sample_values(table, column, k=5)
    Returns a few distinct real values from one column. This is your most
    important grounding tool. The schema tells you a column EXISTS; it does not
    tell you how values are SPELLED or FORMATTED. Before you put any literal in a
    WHERE clause, check the actual stored form — a question may say "USA" while
    the data stores "United States", or say "students" while a status column
    stores "Y"/"N", or use a name whose casing/spacing differs from the data.
    Inspect the column first, then filter on what is really there.

- execute_sql(sql)
    Runs a query against the database and returns the result rows (truncated) or
    an error message. This is for TESTING — run your candidate query here and
    look at the output before you commit to it. If it errors, read the message,
    fix the query, and try again. If the rows look wrong (empty when you expected
    data, wrong count, wrong columns), diagnose and revise rather than submitting
    a guess.

- submit(sql)
    Commits your final SQL as THE answer and ends the task. Call it exactly once,
    only after execute_sql has shown you the query runs and returns sensible
    results. The argument must be the complete, standalone query.

# Process — follow these steps in order

1. ORIENT. Call list_tables() to see what tables exist.

2. MAP. Call get_schema() to learn the columns, types, and foreign-key
   relationships. Identify exactly which tables and columns the question needs
   and how to join them.

3. GROUND THE VALUES. For every literal you intend to filter on (any value in a
   WHERE/HAVING clause, e.g. a country, name, category, or status), call
   sample_values on that column to confirm the exact stored spelling and format.
   Skip this only for purely numeric/date comparisons where no literal string is
   involved. This step is what makes you better than a model that just guesses.

4. WRITE. Compose a single SQLite query. Use only tables and columns you have
   seen in the schema. Join via the foreign keys you found. Match string literals
   to the real values from step 3. Add ORDER BY / LIMIT / DISTINCT / aggregation
   only as the question actually requires.

5. TEST. Run the query with execute_sql. Confirm it executes without error and
   the rows plausibly answer the question (right shape, non-empty unless the
   answer truly is empty, sane counts). If anything is off, return to the
   relevant step and fix it — re-check a value, re-read the schema, correct a
   join — then test again.

6. SUBMIT. Once the query runs and the results look correct, call submit(sql)
   with that exact query. Do not submit a query you have not successfully run.

# Rules

- This is SQLite. Use SQLite syntax and functions.
- Return only the columns the question asks for — no more, no fewer.
- Prefer the simplest query that is correct; do not add filters the question
  does not call for, and do not drop filters it does call for.
- Never finish without calling submit. An answer that is only described in text,
  or left in an execute_sql call, does not count.
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
