"""
Phase-0 harness sanity check (LOCAL, no GPU). Run before trusting any numbers.

Three independent checks, gated so the offline ones always run:

  A. COMPARATOR  — execution_match / is_degenerate on hand-built cases with known
                   answers (wrong query -> False, reordered-equivalent -> True,
                   ORDER BY sensitivity, degenerate guards). No network.
  B. TOOLS       — list_tables / get_schema / sample_values / execute_sql against
                   a real Spider db (concert_singer). No network.
  C. EPISODE     — one full agent episode end-to-end against the TEACHER
                   (DeepSeek, OpenAI-compatible), then score the submitted SQL.
                   Needs DEEPSEEK_API_KEY; skipped with a notice if absent.

Usage:
    .venv/Scripts/python.exe scripts/sanity_check.py            # all three
    .venv/Scripts/python.exe scripts/sanity_check.py --no-teacher   # A + B only
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.db import get_connection, run_query
from src.eval import execution_match, is_degenerate
from src.tools import list_tables, get_schema, sample_values, execute_sql, TOOL_DEFS
from src.agent import run_episode, SYSTEM_PROMPT

DB = "concert_singer"  # a small, present dev database


def _check(label, got, expected):
    ok = got == expected
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: got={got!r} expected={expected!r}")
    return ok


# --- A. Comparator ----------------------------------------------------------

def check_comparator():
    print("\n[A] COMPARATOR - execution_match / is_degenerate")
    passed = []

    # Wrong answer must be False: two different counts on the same db.
    passed.append(_check(
        "wrong query -> False",
        execution_match(DB, "SELECT count(*) FROM singer", "SELECT count(*) FROM stadium"),
        False,
    ))

    # Identical result, rows emitted in a DIFFERENT order. Gold has no ORDER BY,
    # so comparison is order-insensitive -> must be True.
    passed.append(_check(
        "reordered-but-equivalent (no ORDER BY) -> True",
        execution_match(
            DB,
            "SELECT name FROM singer ORDER BY name DESC",   # pred
            "SELECT name FROM singer",                       # gold (unordered)
        ),
        True,
    ))

    # ORDER BY sensitivity: when the gold is ordered, a wrong order must be False.
    # Deterministic, content-independent (constants), so it can't flake.
    gold_ordered = "SELECT 1 AS x UNION ALL SELECT 2 ORDER BY x DESC"   # -> 2,1
    passed.append(_check(
        "ORDER BY, wrong direction -> False",
        execution_match(DB, "SELECT 1 AS x UNION ALL SELECT 2 ORDER BY x ASC",
                        gold_ordered, gold_has_order_by=True),
        False,
    ))
    passed.append(_check(
        "ORDER BY, right direction -> True",
        execution_match(DB, "SELECT 1 AS x UNION ALL SELECT 2 ORDER BY x DESC",
                        gold_ordered, gold_has_order_by=True),
        True,
    ))

    # Degenerate guards: empty result and bare SELECT * are too weak to train on.
    passed.append(_check("is_degenerate(empty) -> 'empty_result'",
                         is_degenerate([], "SELECT name FROM singer WHERE 1=0"),
                         "empty_result"))
    passed.append(_check("is_degenerate(SELECT *) -> 'select_star'",
                         is_degenerate([("a",)], "SELECT * FROM singer"),
                         "select_star"))

    print(f"  -> {sum(passed)}/{len(passed)} comparator checks passed")
    return all(passed)


# --- B. Tools ---------------------------------------------------------------

def check_tools():
    print(f"\n[B] TOOLS - exercised against db '{DB}'")
    ok = True

    tables = list_tables(DB)
    print(f"  list_tables -> {tables}")
    ok &= isinstance(tables, list) and "singer" in tables

    schema = get_schema(DB, tables=["singer"])
    print(f"  get_schema(['singer']) -> {schema.splitlines()[0]} ...")
    ok &= "CREATE TABLE" in schema.upper() and "singer" in schema.lower()

    vals = sample_values(DB, "singer", "country")
    print(f"  sample_values(singer, country) -> {vals}")
    ok &= isinstance(vals, list)

    res = execute_sql(DB, "SELECT count(*) FROM singer")
    print(f"  execute_sql('SELECT count(*) FROM singer') -> {res}")
    ok &= isinstance(res, dict) and res.get("row_count") == 1

    err = execute_sql(DB, "SELECT * FROM no_such_table")
    print(f"  execute_sql(bad sql) -> {err}")
    ok &= isinstance(err, str) and err.startswith("ERROR")

    print(f"  -> tools {'OK' if ok else 'PROBLEM'}")
    return bool(ok)


# --- C. Live teacher episode ------------------------------------------------

def _serialize(msg):
    """Render an assistant message (SDK object or dict) compactly for printing."""
    if isinstance(msg, dict):
        role = msg.get("role")
        calls = msg.get("tool_calls")
        content = msg.get("content")
    else:
        role = getattr(msg, "role", "?")
        calls = getattr(msg, "tool_calls", None)
        content = getattr(msg, "content", None)
    if calls:
        names = ", ".join(
            f"{c.function.name}({c.function.arguments})" if not isinstance(c, dict)
            else f"{c['function']['name']}({c['function']['arguments']})"
            for c in calls
        )
        return f"{role}: CALL {names}"
    return f"{role}: {content}"


def check_episode():
    print("\n[C] EPISODE - one full run against the TEACHER (DeepSeek)")
    try:
        from dotenv import load_dotenv
        load_dotenv()
        from openai import OpenAI
    except ImportError as e:
        print(f"  SKIP: missing dependency ({e})")
        return None

    key = (os.environ.get("DEEPSEEK_API_KEY") or "").strip().strip('"')
    if not key:
        print("  SKIP: DEEPSEEK_API_KEY not set in .env")
        return None

    client = OpenAI(base_url=config.TEACHER_BASE_URL, api_key=key)

    question = "How many singers do we have?"
    gold = "SELECT count(*) FROM singer"
    print(f"  db={DB}  question={question!r}")

    result = run_episode(client, config.TEACHER_MODEL, DB, question, TOOL_DEFS,
                         max_steps=config.MAX_STEPS)

    print("  --- trajectory ---")
    for m in result["messages"]:
        line = _serialize(m)
        print("   ", line if len(line) < 200 else line[:197] + "...")

    pred = result["submitted_sql"]
    print(f"  --- submitted_sql: {pred!r}  (steps={result['steps']}) ---")
    if pred is None:
        print("  [FAIL] teacher never called submit (hit max_steps)")
        return False

    verdict = execution_match(DB, pred, gold)
    print(f"  [{'PASS' if verdict else 'FAIL'}] submitted SQL matches gold (verdict={verdict})")
    return bool(verdict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-teacher", action="store_true",
                        help="Run only the offline checks (A, B).")
    args = parser.parse_args()

    print("Phase-0 harness sanity check")
    print(f"  SYSTEM_PROMPT length: {len(SYSTEM_PROMPT)} chars")

    a = check_comparator()
    b = check_tools()
    c = None if args.no_teacher else check_episode()

    print("\n=== SUMMARY ===")
    print(f"  A comparator: {'PASS' if a else 'FAIL'}")
    print(f"  B tools:      {'PASS' if b else 'FAIL'}")
    print(f"  C episode:    {'PASS' if c else ('SKIP' if c is None else 'FAIL')}")

    # Offline checks must pass; the episode is informational (network/teacher).
    sys.exit(0 if (a and b) else 1)


if __name__ == "__main__":
    main()
