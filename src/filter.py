"""Keep only correct (and non-degenerate) trajectories. The most important step.

Runs each submitted SQL through execution_match and keeps trajectories where:
    - the submitted SQL matches the gold result, AND
    - the match is NOT degenerate (drop empty-result and trivial SELECT * matches).

Clean-and-correct examples are worth far more than raw volume — this filter is
the whole game.

Usage:
    python src/filter.py \
        --in data/trajectories/teacher_raw.jsonl \
        --out data/trajectories/teacher_correct.jsonl

Placeholder — implement in Phase 1.
"""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
