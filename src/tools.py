"""The 5 agent tools. FROZEN after Phase 0 — never change their behavior.

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


def list_tables(db_id):
    raise NotImplementedError


def get_schema(db_id, tables=None):
    raise NotImplementedError


def sample_values(db_id, table, column, k=5):
    raise NotImplementedError


def execute_sql(db_id, sql):
    raise NotImplementedError


def submit(sql):
    raise NotImplementedError


# JSON-schema tool definitions passed to the chat-completions API.
TOOL_DEFS = []  # TODO: fill in function-calling schemas for the 5 tools above
