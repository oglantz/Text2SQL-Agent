"""
Run the agent over a dataset and save transcripts.

Same frozen run_episode loop, but `model` = teacher endpoint (Phase 1) or student
endpoint (Phase 2 rollouts), over train_spider.json / dev.json.

Each output line is one episode:
    {idx, split, db_id, question, gold_sql, submitted_sql, steps, messages, ...}
where `messages` is the full transcript verbatim — the raw material for Phase 1
training. Assistant turns come back from the OpenAI SDK as objects; we serialize
them to plain dicts here (the frozen agent must not be changed to do it).

Robustness for long, paid batches:
    - per-episode try/except with retries, so one bad API call can't kill the run;
    - incremental append + flush, so a crash never loses completed work;
    - --resume to skip episodes already in the output file (keyed by idx).

Usage:
    python src/generate_trajectories.py \
        --split train --teacher deepseek-chat --limit 2000 \
        --out data/trajectories/teacher_raw.jsonl

    # resume an interrupted run (same --out)
    python src/generate_trajectories.py --split train --limit 2000 \
        --out data/trajectories/teacher_raw.jsonl --resume
"""
import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.agent import run_episode
from src.tools import TOOL_DEFS

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional; degrade to a plain iterator
    def tqdm(it, **kwargs):
        return it


# --- dataset ----------------------------------------------------------------

def load_examples(split):
    """Load Spider examples into uniform dicts: {db_id, question, gold_sql}."""
    path = config.TRAIN_JSON if split == "train" else config.DEV_JSON
    if not path.is_file():
        raise FileNotFoundError(f"Spider {split} file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        {"db_id": e["db_id"], "question": e["question"], "gold_sql": e["query"]}
        for e in data
    ]


# --- serialization ----------------------------------------------------------

def serialize_message(msg):
    """Convert one transcript turn to a JSON-serializable dict.

    System/user/tool turns are already plain dicts (the agent appends them as
    such). Assistant turns are OpenAI SDK pydantic objects — `model_dump` gives a
    clean dict preserving content + tool_calls; exclude_none drops the dozens of
    irrelevant null fields (refusal, audio, function_call, ...).
    """
    if isinstance(msg, dict):
        return msg
    if hasattr(msg, "model_dump"):
        return msg.model_dump(exclude_none=True)
    # Last-resort fallback for an unexpected object shape.
    return {
        "role": getattr(msg, "role", "assistant"),
        "content": getattr(msg, "content", None),
    }


def serialize_trajectory(messages):
    return [serialize_message(m) for m in messages]


# --- client -----------------------------------------------------------------

def make_client(base_url, api_key):
    from openai import OpenAI
    return OpenAI(base_url=base_url, api_key=api_key)


def resolve_api_key(env_name):
    """Read the API key from the named env var, tolerating the quotes/whitespace
    that creep into .env files (e.g. `KEY= "sk-..."`)."""
    raw = os.environ.get(env_name, "")
    return raw.strip().strip('"').strip("'").strip()


# --- generation -------------------------------------------------------------

def run_with_retries(client, model, db_id, question, max_steps, temperature,
                     supports_temp, retries=3, backoff=2.0):
    """Call run_episode, retrying transient API/network errors. Raises the last
    error if all attempts fail (caller records it as a failed episode)."""
    kwargs = {"max_steps": max_steps}
    if supports_temp:
        kwargs["temperature"] = temperature
    last_err = None
    for attempt in range(retries):
        try:
            return run_episode(client, model, db_id, question, TOOL_DEFS, **kwargs)
        except Exception as e:  # noqa: BLE001 — broad on purpose; we retry then record
            last_err = e
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
    raise last_err


def load_done_indices(out_path):
    """Indices already present in an existing output file, for --resume."""
    done = set()
    if not Path(out_path).is_file():
        return done
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["idx"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def main():
    parser = argparse.ArgumentParser(description="Generate agent trajectories over Spider.")
    parser.add_argument("--split", choices=["train", "dev"], default="train")
    parser.add_argument("--teacher", default=config.TEACHER_MODEL,
                        help="Model name to send to the endpoint (default: config.TEACHER_MODEL).")
    parser.add_argument("--base-url", default=config.TEACHER_BASE_URL,
                        help="OpenAI-compatible endpoint (default: config.TEACHER_BASE_URL).")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY",
                        help="Env var holding the API key (default: DEEPSEEK_API_KEY).")
    parser.add_argument("--limit", type=int, default=2000,
                        help="Max examples from the start of the split (0 = all).")
    parser.add_argument("--start", type=int, default=0,
                        help="Skip the first N examples (for sharding).")
    parser.add_argument("--temperature", type=float, default=config.EVAL_TEMPERATURE,
                        help="Sampling temperature. NOTE: only honored if the frozen "
                             "run_episode accepts it; otherwise it runs at 0.0.")
    parser.add_argument("--max-steps", type=int, default=config.MAX_STEPS)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume", action="store_true",
                        help="Append to --out, skipping episodes already recorded.")
    args = parser.parse_args()

    # Load .env so the key is available without exporting it manually.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = resolve_api_key(args.api_key_env)
    if not api_key:
        parser.error(f"No API key found in env var {args.api_key_env!r}. "
                     "Set it in .env or the environment.")
    client = make_client(args.base_url, api_key)

    # The frozen loop currently hardcodes temperature=0.0. Honor a non-zero
    # request only if a future run_episode exposes the kwarg; otherwise warn
    # loudly rather than silently producing deterministic rollouts.
    supports_temp = "temperature" in inspect.signature(run_episode).parameters
    if not supports_temp and args.temperature != 0.0:
        print(f"WARNING: run_episode does not accept `temperature`; the frozen loop "
              f"runs at 0.0, ignoring --temperature {args.temperature}.", file=sys.stderr)

    examples = load_examples(args.split)
    examples = examples[args.start:]
    if args.limit and args.limit > 0:
        examples = examples[:args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_indices(out_path) if args.resume else set()
    mode = "a" if args.resume else "w"

    n_total = len(examples)
    n_submitted = n_failed = n_skipped = 0

    print(f"Generating {n_total} {args.split} trajectories with model={args.teacher!r} "
          f"@ {args.base_url}")
    print(f"  out={out_path}  resume={args.resume}  already_done={len(done)}  "
          f"temperature={'(frozen 0.0)' if not supports_temp else args.temperature}")
    print(f"  TIP: watch your provider's usage dashboard for the first ~50 calls to "
          f"sanity-check per-trajectory cost before trusting the full batch.")

    with open(out_path, mode, encoding="utf-8") as f:
        for offset, ex in enumerate(tqdm(examples, desc="episodes", unit="ep")):
            idx = args.start + offset
            if idx in done:
                n_skipped += 1
                continue

            record = {
                "idx": idx,
                "split": args.split,
                "db_id": ex["db_id"],
                "question": ex["question"],
                "gold_sql": ex["gold_sql"],
                "teacher": args.teacher,
            }
            try:
                result = run_with_retries(
                    client, args.teacher, ex["db_id"], ex["question"],
                    args.max_steps, args.temperature, supports_temp,
                )
                record["submitted_sql"] = result["submitted_sql"]
                record["steps"] = result["steps"]
                record["messages"] = serialize_trajectory(result["messages"])
                if result["submitted_sql"]:
                    n_submitted += 1
            except Exception as e:  # noqa: BLE001 — record and move on
                record["submitted_sql"] = None
                record["steps"] = None
                record["messages"] = []
                record["error"] = f"{type(e).__name__}: {e}"
                n_failed += 1

            f.write(json.dumps(record) + "\n")
            f.flush()  # durability: never lose a completed (paid) episode to a crash

    n_written = n_total - n_skipped
    print(f"\nDone. wrote={n_written}  submitted={n_submitted}  "
          f"failed={n_failed}  skipped(resume)={n_skipped}")
    if n_written:
        print(f"  submit rate: {n_submitted}/{n_written} = {n_submitted / n_written:.1%}")
    print(f"  -> {out_path}")
    print(f"  next: python src/filter.py --in {out_path} "
          f"--out data/trajectories/teacher_correct.jsonl --reject-degenerate")


if __name__ == "__main__":
    main()
