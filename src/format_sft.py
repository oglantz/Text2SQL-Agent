"""
Convert filtered trajectories -> HF "messages" training JSONL.

Each output line is exactly {"messages": [...]} in the format TRL's SFTTrainer
consumes for conversational data: a list of {role, content[, tool_calls]} turns,
preserving tool-call and tool-result turns and ending with the model's `submit`.
The model is trained on the WHOLE successful trajectory (system/user/tool context
+ every assistant turn), so the student learns agentic behavior
(explore -> sample -> query -> submit), not just final-SQL generation.

Two format details that matter and are easy to get wrong:

  * arguments -> dict. Trajectories store each tool call the way the OpenAI SDK
    emits it: `function.arguments` is a JSON *string*. Chat templates (Qwen2.5's
    included) render tool calls with `arguments | tojson`, so a string argument is
    double-encoded into a quoted blob. We json.loads it back into a dict so the
    template serializes it once, correctly.
  * assistant content. `model_dump(exclude_none=True)` drops `content` on a
    tool-calling assistant turn; we restore it as "" so every assistant turn has a
    content field (some templates assume one).

Also reports the tokenized length distribution so `max_seq_length` can be set
without the silent truncation that quietly corrupts training. With `transformers`
installed it tokenizes with the real student chat template (the accurate path);
otherwise it falls back to a clearly-labeled character-based approximation and
warns you to confirm on a machine that has the tokenizer (e.g. Colab) before
trusting the number.

Usage:
    python src/format_sft.py \
        --in data/trajectories/teacher_correct.jsonl \
        --out data/sft/train.jsonl \
        [--tokenizer Qwen/Qwen2.5-Coder-7B-Instruct] \
        [--max-seq-length 4096]
"""
import sys
import json
import math
import argparse
from pathlib import Path

# Make `python src/format_sft.py` work as well as `python -m src.format_sft`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import config
except ImportError:  # config is optional; CLI defaults degrade gracefully
    config = None

# The tools the agent was given at generation time. Rendering the length estimate
# WITH them matches what the model sees at inference (the serving endpoint passes
# the same tool schemas), and adds a fixed per-example overhead we want counted.
try:
    from src.tools import TOOL_DEFS
except Exception:  # noqa: BLE001 — tools import shouldn't block formatting
    TOOL_DEFS = None


# --- trajectory -> messages --------------------------------------------------

def normalize_tool_call(tc):
    """One OpenAI-shaped tool call -> chat-template-ready dict.

    Parses the JSON-string `arguments` into a dict (see module docstring); keeps
    the raw string only if it is unparseable, so we never silently lose a call.
    """
    fn = tc.get("function", {}) or {}
    raw_args = fn.get("arguments")
    if isinstance(raw_args, str):
        try:
            arguments = json.loads(raw_args)
        except (json.JSONDecodeError, TypeError):
            arguments = raw_args  # malformed: pass through rather than drop the call
    else:
        arguments = raw_args  # already a dict (some endpoints pre-parse)

    out = {
        "type": tc.get("type", "function"),
        "function": {"name": fn.get("name"), "arguments": arguments},
    }
    if tc.get("id"):
        out["id"] = tc["id"]  # harmless for Qwen, required by some other templates
    return out


def normalize_messages(raw_messages):
    """Normalize a stored transcript into clean, template-ready turns.

    System/user pass through. Tool turns keep their `tool_call_id`. Assistant
    turns get a guaranteed string `content` and, if present, normalized
    `tool_calls`.
    """
    out = []
    for m in raw_messages:
        role = m.get("role")
        if role == "assistant":
            turn = {"role": "assistant", "content": m.get("content") or ""}
            tool_calls = m.get("tool_calls")
            if tool_calls:
                turn["tool_calls"] = [normalize_tool_call(tc) for tc in tool_calls]
            out.append(turn)
        elif role == "tool":
            content = m.get("content")
            out.append({
                "role": "tool",
                "tool_call_id": m.get("tool_call_id"),
                "content": content if content is not None else "",
            })
        elif role in ("system", "user"):
            out.append({"role": role, "content": m.get("content") or ""})
        # Unknown roles are dropped intentionally (nothing else should appear).
    return out


def _has_submit(messages):
    """True if some assistant turn calls `submit` — i.e. the episode actually
    ended in an answer. Post-filter every kept trajectory should; we check
    defensively so a malformed line can't slip a no-answer transcript into SFT."""
    for m in messages:
        for tc in (m.get("tool_calls") or []):
            if (tc.get("function") or {}).get("name") == "submit":
                return True
    return False


def build_example(record):
    """Trajectory record -> ({"messages": [...]}, None) or (None, reason)."""
    raw = record.get("messages")
    if not raw:
        return None, "no_messages"
    messages = normalize_messages(raw)
    if not messages:
        return None, "empty_after_normalize"
    if not _has_submit(messages):
        return None, "no_submit"
    return {"messages": messages}, None


# --- token-length estimation -------------------------------------------------

def load_tokenizer(name):
    """Return (tokenizer, None) on success or (None, reason) so the caller can
    fall back to the approximation. Never raises."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None, "transformers not installed (local env is intentionally light)"
    try:
        return AutoTokenizer.from_pretrained(name), None
    except Exception as e:  # noqa: BLE001 — offline / missing files / etc.
        return None, f"could not load tokenizer {name!r}: {type(e).__name__}: {e}"


def exact_token_count(tokenizer, messages, tools):
    """Length under the model's real chat template. Tries WITH tools (matches
    inference); falls back to without if this template/version rejects them."""
    try:
        ids = tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=True, add_generation_prompt=False,
        )
    except Exception:  # noqa: BLE001 — older templates may not accept `tools`
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
        )
    return len(ids)


def _approx_render(messages, tools):
    """Cheap text rendering used only by the heuristic fallback."""
    parts = []
    if tools:
        parts.append(json.dumps(tools))  # tool schema rides in the prompt every time
    for m in messages:
        parts.append(str(m.get("role")))
        parts.append(str(m.get("content") or ""))
        for tc in (m.get("tool_calls") or []):
            parts.append(json.dumps(tc.get("function") or {}))
    return "\n".join(parts)


def approx_token_count(messages, tools):
    """~chars/4 heuristic. Order-of-magnitude only — labeled as such in the report."""
    return max(1, len(_approx_render(messages, tools)) // 4)


# --- distribution report -----------------------------------------------------

def _percentile(sorted_vals, q):
    if not sorted_vals:
        return 0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = (len(sorted_vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def summarize_lengths(lengths, exact, max_seq_length=None):
    """Print the token-length distribution and a suggested max_seq_length.
    Returns the summary dict (handy for tests / programmatic use)."""
    if not lengths:
        print("No examples to summarize.")
        return {}

    s = sorted(lengths)
    n = len(s)
    pcts = {f"p{int(q * 100)}": _percentile(s, q) for q in (0.5, 0.9, 0.95, 0.99)}
    summary = {
        "n": n, "exact": exact,
        "min": s[0], "max": s[-1],
        "mean": sum(s) / n, **pcts,
    }

    kind = "EXACT (chat template)" if exact else "APPROXIMATE (~chars/4)"
    print(f"\n=== Token length distribution - {kind}, n={n} ===")
    print(f"  min / mean / max : {s[0]} / {summary['mean']:.0f} / {s[-1]}")
    print(f"  p50 / p90 / p95 / p99 : "
          f"{pcts['p50']:.0f} / {pcts['p90']:.0f} / {pcts['p95']:.0f} / {pcts['p99']:.0f}")

    print("  truncation if max_seq_length were:")
    for cap in (1024, 2048, 4096, 8192):
        over = sum(1 for v in s if v > cap)
        mark = "  <- covers all" if over == 0 else ""
        print(f"    {cap:>5}: {over:>5} examples truncated ({over / n:5.1%}){mark}")

    # Suggested cap: smallest multiple of 512 that covers p99, and the one that
    # covers every example (zero truncation). Round up so we never clip the tail.
    def round_up_512(x):
        return int(math.ceil(x / 512.0) * 512)

    suggest_p99 = round_up_512(pcts["p99"])
    suggest_all = round_up_512(s[-1])
    summary["suggested_max_seq_length_p99"] = suggest_p99
    summary["suggested_max_seq_length_all"] = suggest_all
    print(f"  suggested max_seq_length: {suggest_p99} (covers p99) | "
          f"{suggest_all} (covers all {n})")

    if max_seq_length is not None:
        over = sum(1 for v in s if v > max_seq_length)
        summary["over_max_seq_length"] = over
        print(f"  >>> at --max-seq-length {max_seq_length}: {over} ({over / n:.1%}) "
              f"examples would be TRUNCATED.")
        if over:
            print("      Those trajectories lose their tail (often the final submit) "
                  "-> corrupted targets. Raise the cap or drop them.")

    if not exact:
        print("  NOTE: approximate counts. Re-run where `transformers` + the Qwen "
              "tokenizer are available (e.g. Colab) before locking max_seq_length.")
    return summary


# --- driver ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Format filtered trajectories into SFT messages JSONL and "
                    "report the token-length distribution."
    )
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", required=True)
    default_tok = getattr(config, "STUDENT_MODEL", None) if config else None
    parser.add_argument(
        "--tokenizer", default=default_tok,
        help="HF model/tokenizer name for the EXACT length count "
             "(default: config.STUDENT_MODEL).",
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=None,
        help="If set, report how many examples would be truncated at this cap.",
    )
    args = parser.parse_args()

    with open(args.in_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    examples = []
    skipped = {}
    for rec in records:
        ex, reason = build_example(rec)
        if ex is None:
            skipped[reason] = skipped.get(reason, 0) + 1
        else:
            examples.append(ex)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Read {len(records)} trajectories from {args.in_path}")
    print(f"  wrote {len(examples)} SFT examples -> {out_path}")
    if skipped:
        detail = ", ".join(f"{k}={v}" for k, v in sorted(skipped.items()))
        print(f"  skipped {sum(skipped.values())} ({detail})")

    # --- length distribution ---
    tokenizer = exact = None
    if args.tokenizer:
        tokenizer, err = load_tokenizer(args.tokenizer)
        if tokenizer is None:
            print(f"  (length estimate falling back to approximation: {err})")
    else:
        print("  (no --tokenizer / config.STUDENT_MODEL; using approximation)")

    exact = tokenizer is not None
    lengths = []
    for ex in examples:
        msgs = ex["messages"]
        lengths.append(
            exact_token_count(tokenizer, msgs, TOOL_DEFS) if exact
            else approx_token_count(msgs, TOOL_DEFS)
        )

    summarize_lengths(lengths, exact, args.max_seq_length)


if __name__ == "__main__":
    main()
