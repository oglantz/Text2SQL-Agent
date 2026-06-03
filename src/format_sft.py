"""Convert filtered trajectories -> HF "messages" training JSONL.

Each line: {"messages": [...]} in the format SFTTrainer expects, preserving
tool-call and tool-result turns, ending with the model's submit. The model is
trained to predict the assistant turns given system/user/tool context — the WHOLE
successful trajectory is the target, so the student learns agentic behavior
(explore -> sample -> query -> submit), not just final-SQL generation.

Also reports the tokenized length distribution so max_seq_length can be set
without silent truncation.

Usage:
    python src/format_sft.py \
        --in data/trajectories/teacher_correct.jsonl \
        --out data/sft/train.jsonl

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
