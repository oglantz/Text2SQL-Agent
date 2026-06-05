"""
SQLite connection + safe execution helpers.

Contract (Phase 0):
    get_connection(db_id) -> sqlite3.Connection
        Opens data/spider/.../database/<db_id>/<db_id>.sqlite read-only.
    run_query(conn, sql, timeout_s=5) -> (rows, error)
        Executes SQL with a timeout. Returns (rows, None) on success or
        (None, error_str) on failure. A timeout matters: a model will
        eventually generate a cartesian-product query that hangs forever.

Placeholder — implement in Phase 0.
"""


def get_connection(db_id):
    raise NotImplementedError


def run_query(conn, sql, timeout_s=5):
    raise NotImplementedError
