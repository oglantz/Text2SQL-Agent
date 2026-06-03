"""Execution-accuracy comparator — the verifier AND the reward function.
FROZEN after Phase 0 — never change its behavior.

Takes a predicted SQL and the gold SQL, runs both, decides if they match.

    normalize(rows)        -> order-insensitive comparable structure
    execution_match(db_id, pred_sql, gold_sql, gold_has_order_by=False)
        -> True / False / None (None = gold errored; drop the example)

False-positive guard (§4.1): two different queries can return the same table by
accident (both empty, both a single number). Discard degenerate matches when
building training data. Use the official Spider test-suite script (behind a
--official flag) for reported numbers; the fast in-process match for iteration.

Placeholder — implement in Phase 0.
"""
from src.db import get_connection, run_query


def normalize(rows):
    # Order-insensitive comparison unless the query has ORDER BY.
    return sorted([tuple(str(c) for c in row) for row in rows])


def execution_match(db_id, pred_sql, gold_sql, gold_has_order_by=False):
    raise NotImplementedError


def is_degenerate(rows, sql):
    """True if a match here is too easy to trust (empty set, bare SELECT *)."""
    raise NotImplementedError
