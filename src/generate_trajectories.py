"""
Run the agent over a dataset and save transcripts.

Same run_episode loop, but model = teacher endpoint (Phase 1) or student endpoint
(Phase 2 rollouts), over train_spider.json / dev.json.

Usage:
    python src/generate_trajectories.py \
        --split train --teacher deepseek-chat --limit 2000 \
        --out data/trajectories/teacher_raw.jsonl

Each output line: question, db_id, gold_sql, full `messages` transcript,
submitted SQL, step count.

Placeholder — implement in Phase 1.
"""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "dev"], default="train")
    parser.add_argument("--teacher", default="deepseek-chat")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
