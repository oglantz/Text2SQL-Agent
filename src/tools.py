"""
The 5 agent tools. FROZEN after Phase 0 — never change their behavior.

These are exposed to the model as tool/function definitions (JSON schemas) via the
chat-completions tool-calling API. The loop executes the chosen tool and returns
the result as a tool message.

    list_tables(db_id)                 -> [str]    cheap orientation
    get_schema(db_id, tables=None)     -> str      DDL: columns, types, FKs
    sample_values(db_id, table, col)   -> [value]  the value-grounding fix
    execute_sql(db_id, sql)            -> rows|err  agent's main tool + verifier
    submit(sql)                        -> sql       ends the episode

Placeholder — implement in Phase 0.
"""

import json
from src.db import get_connection, run_query

MAX_ROWS = 30

def list_tables(db_id):
    conn = get_connection(db_id)
    rows, err = run_query(conn, "SELECT name FROM sqlite_master WHERE type='table'")
    return f"ERROR: {err}" if err else [r[0] for r in rows]

def get_schema(db_id, tables=None):
    conn = get_connection(db_id)
    rows, err = run_query(conn, "SELECT name, sql FROM sqlite_master WHERE type='table'")
    if err:
        return f"ERROR: {err}"
    # `sql` here is the original CREATE TABLE statement — columns, types, and FKs, already formatted
    return "\n\n".join(create for name, create in rows if not tables or name in tables)


def sample_values(db_id, table, column, k=5):
    conn = get_connection(db_id)
    # Identifiers (table/column) CANNOT be parameterized — only values can.
    # So validate them against the real schema before interpolating (see design note below).
    rows, err = run_query(conn, f'SELECT DISTINCT "{column}" FROM "{table}" LIMIT {int(k)}')
    return f"ERROR: {err}" if err else [r[0] for r in rows]


def execute_sql(db_id, sql):
    conn = get_connection(db_id)
    rows, err = run_query(conn, sql)          # run_query enforces a timeout
    if err:
        return f"ERROR: {err}"                # return the error AS the observation, never raise
    out = {"row_count": len(rows), "rows": rows[:MAX_ROWS]}
    if len(rows) > MAX_ROWS:
        out["note"] = f"showing first {MAX_ROWS} of {len(rows)} rows"
    return out


def dispatch_tool(name, db_id, args):
    if name == "list_tables":   return list_tables(db_id)
    if name == "get_schema":    return get_schema(db_id, tables=args.get("tables"))
    if name == "sample_values": return sample_values(db_id, args["table"], args["column"], k=args.get("k", 5))
    if name == "execute_sql":   return execute_sql(db_id, args["sql"])
    return f"ERROR: unknown tool '{name}'"   # model hallucinated a tool — let it recover, don't crash

# JSON-schema tool definitions passed to the chat-completions API.
TOOL_DEFS = [
    {"type": "function", "function": {
        "name": "list_tables",
        "description": "List all table names in the database. Call this first to orient yourself.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "get_schema",
        "description": "Return CREATE TABLE statements (columns, types, foreign keys) for the given "
                       "tables, or all tables if omitted. Read this before writing SQL.",
        "parameters": {"type": "object", "properties": {
            "tables": {"type": "array", "items": {"type": "string"},
                       "description": "Optional list of table names to restrict to."},
        }, "required": []},
    }},
    {"type": "function", "function": {
        "name": "sample_values",
        "description": "Return a few distinct example values from a column. Use this to check the EXACT "
                       "stored value before writing a WHERE clause (e.g. is it 'United States' or 'USA'?).",
        "parameters": {"type": "object", "properties": {
            "table":  {"type": "string", "description": "Table name."},
            "column": {"type": "string", "description": "Column name."},
            "k":      {"type": "integer", "description": "How many distinct values.", "default": 5},
        }, "required": ["table", "column"]},
    }},
    {"type": "function", "function": {
        "name": "execute_sql",
        "description": "Run a SQL query and return the rows (truncated) or the error message. "
                       "Test your query with this before submitting.",
        "parameters": {"type": "object", "properties": {
            "sql": {"type": "string", "description": "The SQL query to execute."},
        }, "required": ["sql"]},
    }},
    {"type": "function", "function": {
        "name": "submit",
        "description": "Submit your final SQL query as the answer. Call exactly once when confident. Ends the task.",
        "parameters": {"type": "object", "properties": {
            "sql": {"type": "string", "description": "The final SQL query."},
        }, "required": ["sql"]},
    }},
]
