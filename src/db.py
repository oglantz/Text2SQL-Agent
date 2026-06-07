"""
SQLite connection + safe execution helpers.

Contract (Phase 0):
    get_connection(db_id) -> sqlite3.Connection
        Opens data/spider/.../database/<db_id>/<db_id>.sqlite read-only.
    run_query(conn, sql, timeout_s=5) -> (rows, error)
        Executes SQL with a timeout. Returns (rows, None) on success or
        (None, error_str) on failure. A timeout matters: a model will
        eventually generate a cartesian-product query that hangs forever.

The timeout is enforced with sqlite3's progress handler, not the `timeout=`
connect argument — the latter only bounds *lock* contention, whereas a
cartesian-product SELECT burns CPU while holding no lock. The progress handler
fires every N VM opcodes and aborts the statement once the wall-clock deadline
passes, so it interrupts both execution and row fetching.
"""
import sys
import sqlite3
import time
from pathlib import Path

# Make `python src/db.py` work as well as `python -m src.db`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

# How often (in VM opcodes) the progress handler checks the deadline. Small
# enough to interrupt a runaway query promptly, large enough to be cheap.
_PROGRESS_OPCODES = 1000


def get_connection(db_id):
    """Open <db_id>'s SQLite database read-only.

    Read-only matters twice over: it guarantees the verifier can never mutate the
    gold databases, and any model-generated write turns into a clean error rather
    than corrupting state.
    """
    db_path = config.DATABASE_DIR / db_id / f"{db_id}.sqlite"
    if not db_path.is_file():
        raise FileNotFoundError(f"No SQLite database for db_id={db_id!r} at {db_path}")
    # URI connect with mode=ro opens read-only and fails loudly if the file is
    # missing. as_posix() keeps the URI valid on Windows (file:C:/...).
    uri = f"file:{db_path.as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def run_query(conn, sql, timeout_s=5):
    """Execute `sql` against `conn`, bounded by `timeout_s` seconds of wall clock.

    Returns (rows, None) on success, where rows is a list of tuples. On any
    failure returns (None, error_str) with a typed, human-readable message:
        - "timeout: ..."          the query exceeded timeout_s (e.g. cartesian product)
        - "OperationalError: ..." bad SQL, write on a read-only DB, missing table, etc.
        - "<ErrorType>: ..."      any other sqlite3 error
    """
    use_timeout = timeout_s is not None and timeout_s > 0
    if use_timeout:
        deadline = time.monotonic() + timeout_s

        def _watchdog():
            # Non-zero return aborts the running statement with OperationalError.
            return 1 if time.monotonic() > deadline else 0

        conn.set_progress_handler(_watchdog, _PROGRESS_OPCODES)

    cur = conn.cursor()
    try:
        cur.execute(sql)
        rows = cur.fetchall()
        return rows, None
    except sqlite3.OperationalError as e:
        msg = str(e)
        # An abort triggered by the progress handler surfaces as "interrupted".
        if use_timeout and "interrupted" in msg.lower() and time.monotonic() > deadline:
            return None, f"timeout: query exceeded {timeout_s}s"
        return None, f"OperationalError: {msg}"
    except sqlite3.Error as e:
        return None, f"{type(e).__name__}: {e}"
    finally:
        cur.close()
        if use_timeout:
            conn.set_progress_handler(None, 0)
