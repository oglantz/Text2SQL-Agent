"""
Execution-accuracy comparator — the verifier AND the reward function.
FROZEN after Phase 0 — never change its behavior.

Takes a predicted SQL and the gold SQL, runs both, decides if they match.

    normalize(rows)        -> order-insensitive comparable structure
    execution_match(db_id, pred_sql, gold_sql, gold_has_order_by=False)
        -> True / False / None (None = gold errored; drop the example)

Two scoring paths share that frozen comparator; neither changes its behavior:

    default    — fast in-process `execution_match`, one DB instance. Instant,
                 good enough for iteration and quick checks.
    --official — the official Spider test-suite evaluation script, run against
                 many perturbed DB instances per schema. Collapses the
                 false-positive rate to near zero; use it for reported numbers
                 and the on-policy correctness filter (guide §4.1).

False-positive guard (§4.1): two different queries can return the same table by
accident (both empty, both a single number). `is_degenerate` flags such weak
matches so `filter.py --reject-degenerate` can drop them from training data.

CLI:
    # fast, in-process (default)
    python src/eval.py --pred data/trajectories/teacher_raw.jsonl --label teacher_raw
    # rigorous, official test-suite
    python src/eval.py --pred preds.jsonl --official
"""
import sys
from pathlib import Path

# Make `python src/eval.py` work as well as `python -m src.eval`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import os
import re
import subprocess
import tempfile

from src.db import get_connection, run_query

try:
    import config
except ImportError:  # config is optional; CLI defaults degrade gracefully
    config = None


def normalize(rows):
    # Order-insensitive comparison unless the query has ORDER BY.
    # Convert to a comparable structure: sort rows, stringify cells.
    return sorted([tuple(str(c) for c in row) for row in rows])


def execution_match(db_id, pred_sql, gold_sql, gold_has_order_by=False):
    conn = get_connection(db_id)
    gold_rows, gold_err = run_query(conn, gold_sql)
    pred_rows, pred_err = run_query(conn, pred_sql)

    if gold_err:
        return None
    if pred_err:
        return False
    if gold_has_order_by:
        return [tuple(map(str, r)) for r in pred_rows] == [tuple(map(str, r)) for r in gold_rows]

    return normalize(pred_rows) == normalize(gold_rows)


# --- Degenerate-match guard (§4.1) ------------------------------------------
# A "correct" execution match can still be a coincidence rather than evidence the
# query is right. Such matches are too weak to *train* on (see filter.py). This
# implements the Phase-0 stub; it does NOT change the frozen comparator above.

# Words/patterns in a question that imply the answer needs a filter. Heuristic and
# deliberately tunable — it errs toward only firing on strong constraint signals.
_FILTER_HINT_PATTERN = re.compile(
    r"(\bwith\b|\bwhose\b|\bwhere\b|\bnamed\b|\bcalled\b|"
    r"\bmore than\b|\bless than\b|\bgreater than\b|\bfewer than\b|"
    r"\bat least\b|\bat most\b|\bolder than\b|\byounger than\b|"
    r"\bbefore\b|\bafter\b|\bbetween\b|\bequal to\b|"
    r"\bthat (?:have|has|are|is|were|was)\b|\bhaving\b|"
    r"'[^']+'|\"[^\"]+\"|\b\d{4}\b)",
    re.IGNORECASE,
)


def _has_keyword(sql, keyword):
    return re.search(r"\b" + re.escape(keyword) + r"\b", sql or "", re.IGNORECASE) is not None


def is_degenerate(rows, sql, question=None):
    """Return a short reason string if a correct match here is too weak to trust,
    else None. Categories (guide §4.1):

        "empty_result"  — predicted result set is empty (the most common false
                          positive: two unrelated bugs both returning nothing).
        "select_star"   — a trivial bare `SELECT *` whole-table dump.
        "missing_where" — the question implies a filter but the SQL has no
                          WHERE/HAVING (it ignored the question's constraint).

    Truthy return == degenerate, so `if is_degenerate(...)` reads naturally.
    """
    if not rows:
        return "empty_result"

    sql_l = (sql or "").strip().lower()
    # Bare `SELECT * FROM t` with no filter/join/aggregation — trivial dump.
    if re.match(r"^select\s+\*", sql_l) and not _has_keyword(sql, "where") \
            and not _has_keyword(sql, "join") and not _has_keyword(sql, "group"):
        return "select_star"

    if question and _FILTER_HINT_PATTERN.search(question):
        if not _has_keyword(sql, "where") and not _has_keyword(sql, "having"):
            return "missing_where"

    return None


# --- Scoring entry points ----------------------------------------------------

def _has_order_by(sql):
    return "order by" in (sql or "").lower()


def _one_line(sql):
    return " ".join((sql or "").split())


def load_predictions(path):
    """Load a JSONL of predictions/trajectories into uniform example dicts.

    Accepts either `pred_sql`/`submitted_sql` for the prediction and
    `gold_sql`/`query` for the gold, so it consumes trajectory files directly.
    """
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            examples.append({
                "db_id": rec["db_id"],
                "question": rec.get("question"),
                "gold_sql": rec.get("gold_sql") or rec.get("query"),
                "pred_sql": rec.get("pred_sql") or rec.get("submitted_sql"),
                "gold_has_order_by": rec.get("gold_has_order_by"),
            })
    return examples


def score_in_process(examples):
    """Fast in-process scoring via `execution_match`. Gold-errored examples
    (verdict None) are excluded from the denominator, per the Phase-0 contract."""
    n_correct = n_scored = n_gold_errored = 0
    for ex in examples:
        gold_has_order_by = ex.get("gold_has_order_by")
        if gold_has_order_by is None:
            gold_has_order_by = _has_order_by(ex["gold_sql"])
        if not ex.get("pred_sql"):
            verdict = False  # no submission -> wrong
        else:
            verdict = execution_match(ex["db_id"], ex["pred_sql"], ex["gold_sql"], gold_has_order_by)
        if verdict is None:
            n_gold_errored += 1
        else:
            n_scored += 1
            n_correct += 1 if verdict else 0
    accuracy = (n_correct / n_scored) if n_scored else 0.0
    return {
        "accuracy": accuracy,
        "n": len(examples),
        "n_scored": n_scored,
        "n_correct": n_correct,
        "n_gold_errored": n_gold_errored,
    }


def _resolve_test_suite(test_suite_dir=None, db_dir=None, tables_json=None):
    """Resolve official-eval locations: explicit arg > env var > config."""
    def pick(arg, env, cfg_attr):
        if arg:
            return str(arg)
        env_val = os.environ.get(env)
        if env_val:
            return env_val
        if config is not None and getattr(config, cfg_attr, None):
            return str(getattr(config, cfg_attr))
        return ""

    return (
        pick(test_suite_dir, "SPIDER_TEST_SUITE_DIR", "SPIDER_TEST_SUITE_DIR"),
        pick(db_dir, "SPIDER_TEST_SUITE_DB", "TEST_SUITE_DB_DIR"),
        pick(tables_json, "SPIDER_TABLES_JSON", "TABLES_JSON"),
    )


def _parse_execution_accuracy(output):
    # The test-suite report prints a row beginning with "execution" whose last
    # column is the overall ("all") accuracy across difficulty buckets.
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "execution":
            try:
                return float(parts[-1])
            except ValueError:
                continue
    return None


def score_official(examples, test_suite_dir=None, db_dir=None, tables_json=None):
    """Score predictions with the official Spider test-suite evaluation script.

    Writes the parallel gold/pred files the script expects, invokes it as a
    subprocess against the multi-instance test-suite databases, and parses the
    overall execution accuracy from its report.

    Returns {"accuracy": float, "n": int, "raw_output": str}. Raises RuntimeError
    with setup instructions if the script/databases aren't available.
    """
    ts_dir, ts_db, ts_tables = _resolve_test_suite(test_suite_dir, db_dir, tables_json)

    if not ts_dir or not Path(ts_dir).is_dir():
        raise RuntimeError(
            "Official Spider test-suite not found. Set it up once:\n"
            "  git clone https://github.com/taoyds/test-suite-sql-eval "
            "third_party/test-suite-sql-eval\n"
            "then set SPIDER_TEST_SUITE_DIR (env) or config.SPIDER_TEST_SUITE_DIR, and\n"
            "SPIDER_TEST_SUITE_DB to the downloaded test-suite databases directory."
        )
    eval_script = Path(ts_dir) / "evaluation.py"
    if not eval_script.is_file():
        raise RuntimeError(f"evaluation.py not found in {ts_dir}")
    if not ts_db or not Path(ts_db).is_dir():
        raise RuntimeError(
            f"Test-suite database dir not found ({ts_db!r}). Set SPIDER_TEST_SUITE_DB "
            "or config.TEST_SUITE_DB_DIR to the test-suite databases directory."
        )

    # gold file: "<gold_sql>\t<db_id>" per line; pred file: "<pred_sql>" per line.
    tmp = Path(tempfile.mkdtemp(prefix="spider_eval_"))
    gold_path, pred_path = tmp / "gold.txt", tmp / "pred.txt"
    with open(gold_path, "w", encoding="utf-8") as gf, \
            open(pred_path, "w", encoding="utf-8") as pf:
        for ex in examples:
            # A missing prediction must count as wrong; emit a guaranteed-bad SQL
            # rather than a blank line (which the script can choke on).
            pred_sql = _one_line(ex.get("pred_sql")) or "SELECT 0 WHERE 1=2"
            gf.write(f"{_one_line(ex['gold_sql'])}\t{ex['db_id']}\n")
            pf.write(f"{pred_sql}\n")

    cmd = [
        sys.executable, str(eval_script),
        "--gold", str(gold_path),
        "--pred", str(pred_path),
        "--db", str(ts_db),
        "--etype", "exec",
    ]
    if ts_tables:
        cmd += ["--table", str(ts_tables)]

    proc = subprocess.run(cmd, cwd=ts_dir, capture_output=True, text=True)
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        raise RuntimeError(
            f"test-suite evaluation.py failed (exit {proc.returncode}):\n{output}"
        )

    accuracy = _parse_execution_accuracy(output)
    if accuracy is None:
        raise RuntimeError(
            f"Could not parse execution accuracy from script output:\n{output}"
        )
    return {"accuracy": accuracy, "n": len(examples), "raw_output": output}


def main():
    parser = argparse.ArgumentParser(
        description="Score predicted SQL by execution accuracy."
    )
    parser.add_argument("--pred", required=True, help="JSONL of predictions/trajectories.")
    parser.add_argument(
        "--official", action="store_true",
        help="Use the official Spider test-suite script instead of the fast "
             "in-process comparator (the default).",
    )
    parser.add_argument("--label", default=None, help="Label recorded with the result.")
    parser.add_argument(
        "--out", default=None,
        help="Append the result record here (default: config.EVAL_RUNS).",
    )
    parser.add_argument("--test-suite-dir", default=None, help="Official-eval: repo dir.")
    parser.add_argument("--db-dir", default=None, help="Official-eval: test-suite DB dir.")
    parser.add_argument("--tables", default=None, help="Official-eval: tables.json.")
    args = parser.parse_args()

    examples = load_predictions(args.pred)

    if args.official:
        summary = score_official(examples, args.test_suite_dir, args.db_dir, args.tables)
        method = "official_test_suite"
    else:
        summary = score_in_process(examples)
        method = "in_process"

    record = {
        "label": args.label,
        "pred_file": args.pred,
        "method": method,
        "accuracy": round(summary["accuracy"], 4),
        **{k: v for k, v in summary.items() if k not in ("accuracy", "raw_output")},
    }
    print(json.dumps(record, indent=2))

    out_path = args.out
    if out_path is None and config is not None:
        out_path = str(config.EVAL_RUNS)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
