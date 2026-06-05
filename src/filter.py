"""
Keep only correct (and non-degenerate) trajectories. The most important step.

Runs each submitted SQL through execution_match and keeps trajectories where:
    - the submitted SQL matches the gold result, AND
    - (with --reject-degenerate) the match is NOT degenerate (drop empty-result
      and trivial SELECT * / no-WHERE-when-the-question-implies-one matches).

Clean-and-correct examples are worth far more than raw volume — this filter is
the whole game.

Each input line is a trajectory: question, db_id, gold_sql, submitted_sql (and
the full messages transcript, which is passed through untouched).

Usage:
    python src/filter.py \
        --in data/trajectories/teacher_raw.jsonl \
        --out data/trajectories/teacher_correct.jsonl \
        [--reject-degenerate]
"""
import sys
import json
import argparse
from collections import Counter
from pathlib import Path

# Make `python src/filter.py` work as well as `python -m src.filter`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import get_connection, run_query
from src.eval import execution_match, is_degenerate


def _gold_has_order_by(rec):
    val = rec.get("gold_has_order_by")
    if val is not None:
        return val
    return "order by" in (rec.get("gold_sql") or "").lower()


def filter_trajectories(records, reject_degenerate=False):
    """Core filter: returns (kept_records, stats Counter, degenerate_reasons Counter).

    Side-effect free apart from DB reads, so it's unit-testable on hand-built
    records. `stats` tallies every drop reason; `degenerate_reasons` breaks the
    degenerate drops down by category (empty_result / select_star / missing_where).
    """
    kept = []
    stats = Counter()
    degenerate_reasons = Counter()
    for rec in records:
        stats["total"] += 1
        pred_sql = rec.get("submitted_sql") or rec.get("pred_sql")
        gold_sql = rec.get("gold_sql")
        db_id = rec.get("db_id")

        if not pred_sql:
            stats["dropped_no_submit"] += 1
            continue

        verdict = execution_match(db_id, pred_sql, gold_sql, _gold_has_order_by(rec))
        if verdict is None:
            stats["dropped_gold_error"] += 1
            continue
        if not verdict:
            stats["dropped_incorrect"] += 1
            continue

        if reject_degenerate:
            conn = get_connection(db_id)
            pred_rows, pred_err = run_query(conn, pred_sql)
            reason = None if pred_err else is_degenerate(pred_rows, pred_sql, rec.get("question"))
            if reason:
                stats["dropped_degenerate"] += 1
                degenerate_reasons[reason] += 1
                continue

        kept.append(rec)
        stats["kept"] += 1
    return kept, stats, degenerate_reasons


def main():
    parser = argparse.ArgumentParser(
        description="Keep only correct, non-degenerate trajectories."
    )
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--reject-degenerate", action="store_true",
        help="Also drop weak matches: empty result, bare SELECT *, and "
             "no-WHERE-when-the-question-implies-one. Logs how many were dropped.",
    )
    args = parser.parse_args()

    with open(args.in_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    kept, stats, degenerate_reasons = filter_trajectories(records, args.reject_degenerate)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec) + "\n")

    # --- log ---
    print(f"Read {stats['total']} trajectories from {args.in_path}")
    print(f"  kept:               {stats['kept']}")
    print(f"  dropped incorrect:  {stats['dropped_incorrect']}")
    print(f"  dropped no-submit:  {stats['dropped_no_submit']}")
    print(f"  dropped gold-error: {stats['dropped_gold_error']}")
    if args.reject_degenerate:
        line = f"  dropped degenerate: {stats['dropped_degenerate']}"
        if degenerate_reasons:
            breakdown = ", ".join(f"{k}={v}" for k, v in sorted(degenerate_reasons.items()))
            line += f"  ({breakdown})"
        print(line)
    print(f"Wrote {stats['kept']} -> {args.out}")


if __name__ == "__main__":
    main()
