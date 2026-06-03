# Text-to-SQL Agent → Distillation: End-to-End Project Guide

A self-contained learning project that forces you to actually understand three things your manager named: **agents**, **LLM fine-tuning**, and **distillation**. You will build a tool-using SQL agent, harvest a strong model's behavior, and compress that behavior into a small open model — measuring improvement at every step.

This guide over-explains on purpose. Read Section 1 before touching code; it's the part that turns "I ran a script" into "I understand what I did."

---

## Table of Contents

0. [How to use this guide](#0-how-to-use-this-guide)
1. [Conceptual foundations (read this first)](#1-conceptual-foundations)
2. [Architecture and repo layout](#2-architecture-and-repo-layout)
3. [Environment setup](#3-environment-setup)
4. [Phase 0 — Eval harness + agent + baseline (the gate)](#4-phase-0--eval-harness--agent--baseline-the-gate)
5. [Phase 1 — Off-policy distillation (= fine-tuning)](#5-phase-1--off-policy-distillation--fine-tuning)
6. [Phase 2 — On-policy distillation](#6-phase-2--on-policy-distillation)
7. [Phase 3 — Scale up and analyze](#7-phase-3--scale-up-and-analyze)
8. [The traps, consolidated](#8-the-traps-consolidated)
9. [What "done" looks like](#9-what-done-looks-like)
10. [Resources](#10-resources)

---

## 0. How to use this guide

**The one rule that makes the whole project coherent:** the agent loop, the tools, and the eval comparator stay *frozen* across every phase. The **only** thing that changes is the model weights. If you change the harness between phases, your before/after numbers are meaningless and you've learned nothing measurable. Lock the harness in Phase 0 and never touch it again.

**The mental model:** these are not three separate mini-projects. They are one pipeline:

```
strong model acts as an AGENT  →  you harvest its successful trajectories
        →  FINE-TUNE a small model on them (off-policy DISTILLATION)
        →  small model generates its own trajectories, you train on those (on-policy DISTILLATION)
        →  measure execution accuracy after each step
```

Every arrow is one of your three topics. The reason this specific task works is that **the verifier and the reward are the same object**: run the SQL, compare the result table to the gold answer. That single fact gives you (a) a free, instant, objective eval and (b) a free reward signal for on-policy training, with short trajectories that don't spiral into the failure modes that kill harder agent tasks.

**Working with Claude Code:** you'll do implementation in VS Code with the Claude Code CLI. The right division of labor:

- **You** own the *design decisions* and *concepts* — what the tools are, what counts as a correct answer, what data format the trainer needs, why a run failed. This guide arms you for that.
- **Claude Code** owns the *plumbing* — parsing the Spider JSON, wiring argument parsers, writing the file I/O, debugging stack traces.

Concretely: read a section here, understand the *what* and *why*, then hand Claude Code a precise task like *"Write a Python function that loads Spider's `dev.json` and returns a list of `{question, db_id, gold_sql}` dicts; the databases live in `spider/database/<db_id>/<db_id>.sqlite`."* Precise tasks with clear contracts get good code. Vague tasks ("build the agent") get slop you can't debug. The skill you're building is being able to *specify* and *verify* each piece — that's exactly the skill that matters on the job.

**Time budget:** Phase 0–2 on Spider is roughly 2–4 weekends. Phase 3 is a stretch. Do not add a web UI. Do not add features. Depth over surface area.

---

## 1. Conceptual foundations

Skipping this section is the difference between being able to *talk about* what you built and just having run someone's code. Your manager will ask "why," not "did it run."

### 1.1 Agents

An **agent**, in the LLM sense, is a model placed in a loop where it can take **actions** that change its environment and **observe** the results, repeating until it decides it's done. The dominant pattern is **ReAct** (Reason + Act): the model alternates between a private reasoning step and a concrete tool call, feeding each observation back into context.

Contrast with a one-shot model: you ask "write SQL for this question," it emits a guess, done. No feedback, no recovery. An agent instead does:

```
THINK:    "I need to know the schema before I can write this."
ACT:      get_schema()
OBSERVE:  <tables, columns, foreign keys come back>
THINK:    "The question filters on country. Let me check the actual stored values."
ACT:      sample_values(table="customers", column="country")
OBSERVE:  ["United States", "Canada", ...]   ← not "USA"
THINK:    "Now I can write the query with the correct literal."
ACT:      execute_sql("SELECT ... WHERE country = 'United States'")
OBSERVE:  <result rows, or an error message>
THINK:    "Rows look right."
ACT:      submit(query)
```

**Why the agentic framing genuinely matters here, not just as buzzword compliance:** the single biggest source of text-to-SQL errors in practice is *value grounding* — the query needs an exact literal that lives in the data (`'United States'` vs `'USA'` vs `'US'`). A one-shot model has to guess and usually guesses wrong. An agent with a `sample_values` tool *discovers* the real value. This is the cleanest demonstration in the whole project of *why you built an agent at all*, and you should make sure that tool exists and that you point to it in your writeup. The second-biggest error source is *schema linking* (picking the right table/column), which the `get_schema` exploration step addresses the same way.

Your tools for this project:

| Tool | Signature | Purpose |
|---|---|---|
| `list_tables` | `() -> [str]` | Cheap orientation: what tables exist. |
| `get_schema` | `(tables?) -> str` | DDL: columns, types, foreign keys. The agent's map. |
| `sample_values` | `(table, column) -> [value]` | A few distinct values. The value-grounding fix. |
| `execute_sql` | `(query) -> rows \| error` | Run against SQLite. Both the agent's main tool **and** your verifier. |
| `submit` | `(query) -> end` | Commit the final answer; ends the episode. |

A **trajectory** (or rollout) is the full transcript of one episode: every thought, tool call, observation, and the final query. Trajectories are the raw material for everything downstream — they are what you fine-tune on. A trajectory is **successful** if its submitted query's result set matches the gold answer.

### 1.2 Fine-tuning

**Fine-tuning** = continuing to train an already-pretrained model on a narrower dataset so it adapts to your task. You're nudging existing weights, not training from scratch.

**The spectrum of how much you update:**

- **Full fine-tuning** — update all weights. Best quality ceiling, brutal memory cost. A 7B model in full fp32 fine-tuning needs ~10× the model size in VRAM once you count gradients and optimizer state (Adam keeps two extra numbers per parameter). Out of reach on your budget.
- **LoRA (Low-Rank Adaptation)** — *freeze* the original weights and inject small trainable "adapter" matrices alongside them. The insight: the *change* a fine-tune needs to make to a big weight matrix is approximately low-rank, so instead of learning a full `d×d` update you learn two skinny matrices `A` (`d×r`) and `B` (`r×d`) with `r` tiny (8–64). You train maybe 1% of the parameter count. At inference the adapter can be merged back in, so there's no speed penalty.
- **QLoRA (Quantized LoRA)** — same as LoRA, but the frozen base model is also **quantized to 4-bit** (NF4 format) to slash memory further. The adapters stay in higher precision (bf16) so training signal isn't lost. This is what lets a 7B model fine-tune on a single consumer/Colab GPU. **This is what you'll use.**

Rough memory picture (7B-class model), to make the tradeoff concrete:

| Method | Base weights | Adapters | Total VRAM (incl. optimizer) |
|---|---|---|---|
| Full FT | ~28 GB (bf16) | — | ~120–240 GB |
| LoRA | ~14–28 GB | ~0.5 GB | ~30 GB |
| QLoRA | ~7 GB (4-bit) | ~0.5 GB | **~10–16 GB** |

**Key hyperparameters you'll set and should understand:**

- **`r` (rank):** adapter capacity. Higher = more it can learn, more risk of overfitting and more memory. Start at 16 or 32.
- **`lora_alpha`:** a scaling factor on the adapter output. Convention is `alpha = r` or `alpha = 2r`. (The effective scale applied is `alpha/r`.)
- **`target_modules`:** which weight matrices get adapters. Standard practice is all attention projections plus the MLP projections: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`.
- **`lora_dropout`:** regularization; `0` is fine (and is the optimized path in Unsloth) for this much data.
- **learning rate:** LoRA tolerates higher LRs than full FT; `1e-4` to `2e-4` is a typical starting band.

**The thing nobody tells beginners, stated bluntly:** a fine-tune that lowers your *training loss* but doesn't move your *task metric* (execution accuracy) **failed**. Loss going down means the model memorized your trajectories' token patterns; it does not mean it got better at writing correct SQL. You will only know if it worked because you built the eval harness first. This is why Phase 0 is the gate, not Phase 1.

**Data quality dominates quantity.** A few hundred *clean, verified-correct* trajectories beat thousands of noisy ones. Your correctness filter (keep only trajectories whose SQL actually returned the right answer) is not a nice-to-have — it's the single most important determinant of whether the fine-tune works.

### 1.3 Distillation

**Distillation** = transferring capability from a strong "teacher" model into a smaller, cheaper "student." The word covers three genuinely different techniques, and conflating them is the most common beginner mistake. Your manager listed fine-tuning *and* distillation separately, which strongly implies they want you to know the difference. You'll implement the first two and be able to explain the third.

**(1) Sequence-level / "data" distillation (off-policy).**
The teacher generates outputs (here: full trajectories). You filter to the good ones and run ordinary supervised fine-tuning on them. *This is literally just fine-tuning, where the training data happens to be teacher-written.* This is the bridge between your two topics — "distillation" and "fine-tuning" are the same mechanical act here, differing only in where the data came from. Cheap, robust, works with any teacher you can call (even a closed API), because you only need its *text outputs*.

It's called **off-policy** because the training data comes from a different policy (the teacher's) than the student's own. The student learns to imitate transcripts it would never have generated itself.

**(2) On-policy distillation.**
The **student** generates the trajectories (rolls out on the task), and the teacher provides the *correction signal* on the student's own outputs. Two ways to get that signal:

- **Rejection-sampling fine-tuning (the simple, robust form you'll do first):** student rolls out, you keep only the trajectories that pass the execution check, retrain on those. The "teacher" here is effectively the *environment's reward* (did the SQL return the right rows). The student learns from its *own* successful attempts.
- **Token-level on-policy distillation (the advanced form):** student rolls out; a teacher model scores *every token* the student produced via its log-probabilities, and you train the student to match the teacher's per-token distribution (a KL/JSD divergence). This needs a teacher whose logits you can read — so an **open-weights teacher you run yourself**, not a closed API.

Why on-policy is the technique everyone moved to: it trains the model on *its own* mistakes and states, eliminating the train/inference mismatch that plagues pure imitation. A model fine-tuned only on a teacher's perfect transcripts has never seen its *own* half-wrong intermediate states, so it doesn't know how to recover from them at test time. On-policy fixes exactly that.

The catch that's directly relevant to why we chose SQL: in on-policy agent training, **errors cascade** — one wrong tool call poisons every subsequent step, and the longer the trajectory the worse it gets. This is *the* reason we rejected SWE-bench (long horizons, severe cascading) and chose SQL (short horizons, 2–6 steps, cascading stays manageable).

**(3) Classic logit KD (Hinton).**
Match the teacher's full output-logit distribution token-for-token on a fixed dataset. Requires teacher and student to share a tokenizer/vocabulary. Mostly superseded by the above. You don't need to implement it — just be able to say "that's the original 2015 formulation; modern practice uses on-policy generalized-JSD methods because they handle the capacity gap and distribution mismatch better."

**A beautiful practical detail you'll exploit in Phase 2:** Hugging Face TRL's `GKDTrainer` (Generalized Knowledge Distillation) unifies *all* of these under two knobs:
- `lmbda` — the fraction of training data that is *on-policy* (student-generated). `lmbda=0` → off-policy supervised distillation; `lmbda=1` → fully on-policy; in between → a mix.
- `beta` — which divergence: `0` = forward KL, `1` = reverse KL, between = generalized Jensen-Shannon.

So the conceptual distinction you just learned maps *directly* onto a single config parameter. Flipping `lmbda` from 0 to 1 in the same trainer literally is the move from off-policy to on-policy distillation. That's your cleanest possible demonstration that you understand the difference.

### 1.4 How they compose

| Phase | What you build | "Agents" | "Fine-tuning" | "Distillation" |
|---|---|---|---|---|
| 0 | Agent loop + eval + baseline | ✅ the loop itself | — | — (baseline only) |
| 1 | SFT student on teacher trajectories | runs the loop | ✅ QLoRA SFT | ✅ off-policy / sequence-level |
| 2 | Rejection-sampling + token-level OPD | student runs the loop | ✅ retrain | ✅ on-policy |
| 3 | BIRD + error analysis | harder env | — | comparison + failure modes |

The headline result you're working toward is a single rising curve: **base student → +off-policy → +on-policy**, measured in execution accuracy. That curve *is* the project.

---

## 2. Architecture and repo layout

### The local / Colab split (understand this before you set anything up)

You have two execution contexts, and putting the right work in each is what keeps cost near zero:

- **Local (VS Code, your laptop, CPU is fine):** all the *model-agnostic* code — the agent loop, the tools, the eval comparator, data scripts, analysis. Also **teacher trajectory generation**, because that's just API calls. None of this needs a GPU.
- **Colab (GPU):** anything that loads open-model *weights* — i.e., **training** (SFT, retraining, GKD) and **evaluating the open student** (running a 7B for inference needs a GPU; you'll serve it with vLLM).

**The unifying trick that makes this clean:** write your agent loop to talk to an **OpenAI-compatible chat-completions endpoint** with tool calling. Then:
- Teacher = point it at a hosted API (Anthropic / OpenAI / DeepSeek).
- Student = serve your open model with **vLLM** (on Colab), which exposes the *same* OpenAI-compatible API.

Same agent-loop code drives both. You swap a base URL and a model name. Nothing else changes. This is also just good engineering practice and worth being able to explain.

The repo is plain Python in git; you `git clone` it onto Colab so the identical code runs in both contexts.

### Repo layout

```
text2sql-distill/
├── README.md
├── requirements.txt
├── .env.example                 # API keys (never commit real keys)
├── config.py                    # paths, model names, endpoints, hyperparams
├── src/
│   ├── tools.py                 # the 5 tools: schema, sample, execute, submit
│   ├── agent.py                 # the ReAct loop (model-agnostic)
│   ├── db.py                    # SQLite connection + safe execution helpers
│   ├── eval.py                  # execution-accuracy comparator (the verifier)
│   ├── generate_trajectories.py # run agent over a dataset, save transcripts
│   ├── filter.py                # keep only correct (and non-degenerate) trajectories
│   ├── format_sft.py            # trajectories -> HF "messages" training JSONL
│   └── analyze.py               # error taxonomy, plots (Phase 3)
├── notebooks/
│   ├── 01_sft_unsloth.ipynb     # Colab: QLoRA SFT (Phase 1)
│   ├── 02_serve_vllm.ipynb      # Colab: serve a model for eval
│   ├── 03_rejection_sampling.ipynb  # Colab: on-policy round (Phase 2A)
│   └── 04_gkd.ipynb             # Colab: token-level on-policy distill (Phase 2B)
├── data/
│   ├── spider/                  # downloaded dataset (gitignored)
│   ├── trajectories/            # raw + filtered transcripts (gitignored)
│   └── sft/                     # formatted training JSONL (gitignored)
└── results/
    └── eval_runs.jsonl          # every eval result, append-only (your evidence)
```

### Data flow

```
Spider (SQLite DBs + question/gold pairs)
        │
        ▼
[LOCAL] generate_trajectories.py  ──teacher API──►  raw teacher trajectories
        │
        ▼
[LOCAL] filter.py  ──execution check──►  correct trajectories only
        │
        ▼
[LOCAL] format_sft.py  ──►  train.jsonl (HF messages format)
        │
        ▼  (git push / Drive upload)
[COLAB] 01_sft_unsloth.ipynb  ──QLoRA──►  LoRA adapter
        │
        ▼  (download adapter, or eval in-place on Colab)
[COLAB] 02_serve_vllm.ipynb  ──serve student──►  OpenAI-compatible endpoint
        │
        ▼
[either] agent loop + eval.py against student  ──►  execution accuracy → results/
```

---

## 3. Environment setup

### 3.1 Local environment

You need Python 3.11+ and a clean virtual environment. Conda or `venv` both fine.

```bash
# create project + venv
mkdir text2sql-distill && cd text2sql-distill
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# local deps (NO heavy ML libs here — those live on Colab)
pip install --upgrade pip
pip install openai anthropic python-dotenv datasets tqdm pandas matplotlib
```

`requirements.txt` (local):

```
openai
anthropic
python-dotenv
datasets
tqdm
pandas
matplotlib
```

> Why no `torch`/`unsloth`/`trl` locally: you are not running models locally. Keeping the local env light avoids a multi-GB install you'll never use on CPU. The training deps are installed *inside* the Colab notebooks.

### 3.2 VS Code + Claude Code CLI

1. Open the folder in VS Code (`code .`).
2. Install the Python extension; select your `.venv` as the interpreter (Cmd/Ctrl-Shift-P → "Python: Select Interpreter").
3. Install the Claude Code CLI per the official docs and run `claude` from the project root in the integrated terminal. Because the project's structure and conventions live in your repo, Claude Code can read your existing files for context when you ask it to add or fix code.

> The product specifics for installing/configuring Claude Code change over time — check the current docs at `docs.claude.com` rather than trusting any snapshot here.

**A good `CLAUDE.md`** in the repo root pays off — Claude Code reads it for project context. Put the one rule in it:

```markdown
# Project: text-to-SQL agent distillation
- The agent loop, tools, and eval comparator are FROZEN after Phase 0. Never modify their behavior.
- The agent talks to an OpenAI-compatible endpoint; teacher and student differ only by base_url + model name.
- "Correct" = predicted SQL's result set matches gold result set (see src/eval.py). Guard against degenerate matches.
- Style: small, testable functions with explicit input/output contracts.
```

### 3.3 API keys (the teacher)

Copy `.env.example` to `.env` and fill in whichever teacher you use. **Add `.env` to `.gitignore` immediately** — committing a key is the kind of mistake that's expensive and public.

```
# .env  (pick the one you'll use)
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
DEEPSEEK_API_KEY=...
```

**Teacher choice, ranked for this project:**

1. **DeepSeek** — cheapest by a wide margin and strong at code/SQL. Best default for the off-policy phase where you just need lots of short trajectories. Tradeoff: it's a separate account/key.
2. **A small frontier model (Claude Haiku-class / GPT-4o-mini-class)** — slightly pricier, very reliable tool-calling, convenient if you already have the key. 
3. **A larger frontier model** — only if your trajectory yield from the cheaper ones is too low. Don't start here; it's easy to spend $50 you didn't need to.

> For the off-policy phase the teacher only needs to *emit text/tool calls*, so any of these works. For **token-level** on-policy distillation (Phase 2B) you need teacher *logits*, which closed APIs don't give you — that teacher must be an **open-weights model you run on Colab** (e.g. a 14–32B coder model). Plan for that split now.

### 3.4 Colab setup

In each training notebook, first cell installs the heavy stack. Unsloth is the fastest path and fits a 7B QLoRA comfortably; it also patches TRL so the trainers "just work."

```python
# Colab cell 1 — install (versions move fast; pin what works for you)
!pip install -q unsloth trl peft accelerate bitsandbytes datasets vllm

import torch
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
```

**Which Colab GPU you need, honestly:**

- **Free tier (T4, 16 GB):** enough for a **3B** student QLoRA at short sequence lengths, and enough to *serve* a small model for eval. Tight but workable. Good for getting the pipeline running end to end before you spend anything.
- **Colab Pro / Pro+ (L4 / A100 40GB):** what you want for a **7B** student and for comfortable sequence lengths. The A100 is the difference between a training run taking minutes vs. fighting OOM. Spend here, not on the teacher.

**Getting your repo onto Colab** — two options:
- `!git clone https://github.com/<you>/text2sql-distill.git` (cleanest; re-clone to get updates).
- Mount Google Drive and keep the repo there (`from google.colab import drive; drive.mount('/content/drive')`) — convenient for persisting adapters and trajectories between sessions, since Colab wipes local disk.

> Colab disconnects and resets. **Persist anything you care about** (trained adapters, generated trajectories) to Drive or the Hub immediately after creating it. Losing a 40-minute training run to a disconnect is a rite of passage you can skip.

### 3.5 Get the data (Spider)

Spider 1.0 ships as a set of SQLite databases plus JSON files of `(question, db_id, gold_sql)` examples. Download it (Yale's official release / its Hugging Face mirror), unzip into `data/spider/`. You're looking for roughly this structure:

```
data/spider/
├── train_spider.json        # ~7k training examples
├── dev.json                 # ~1k dev examples (your eval set)
├── tables.json              # schema metadata for all DBs
└── database/
    └── <db_id>/<db_id>.sqlite
```

> **Task for Claude Code:** *"Write `src/db.py` with a function `get_connection(db_id)` that opens `data/spider/database/<db_id>/<db_id>.sqlite` read-only, and `run_query(conn, sql, timeout_s=5)` that executes SQL with a timeout and returns either rows or a structured error. Use Python's sqlite3."* A timeout matters — a model will eventually generate a cartesian-product query that hangs forever.

Hold off on BIRD until Phase 3; it's bigger and messier and you don't want its weight while you're still debugging the pipeline.

---

## 4. Phase 0 — Eval harness + agent + baseline (the gate)

**Goal:** a frozen, trustworthy way to measure "what fraction of dev questions does a given model get right," and a baseline number for the untrained student. **If you cannot complete this phase cheaply and reliably, stop — everything downstream is built on it.** This is also where you do all your boring-but-critical debugging while the stakes are low.

### 4.1 The execution-accuracy comparator (build this first, before the agent)

This is your verifier *and* your reward function. It takes a predicted SQL and the gold SQL, runs both, and decides if they match.

```python
# src/eval.py  (core idea — have Claude Code harden it)
from src.db import get_connection, run_query

def normalize(rows):
    # Order-insensitive comparison unless the query has ORDER BY.
    # Convert to a comparable structure: sort rows, stringify cells.
    return sorted([tuple(str(c) for c in row) for row in rows])

def execution_match(db_id, pred_sql, gold_sql, gold_has_order_by=False):
    conn = get_connection(db_id)
    gold_rows, gold_err = run_query(conn, gold_sql)
    pred_rows, pred_err = run_query(conn, pred_sql)
    if gold_err:            # gold should never error; if it does, drop the example
        return None
    if pred_err:
        return False
    if gold_has_order_by:
        return [tuple(map(str, r)) for r in pred_rows] == \
               [tuple(map(str, r)) for r in gold_rows]
    return normalize(pred_rows) == normalize(gold_rows)
```

**The false-positive problem — internalize this, it bites in Phase 2.** Two *different* queries can return the *same* table by accident: both empty, both a single number, both one row. A spuriously-"correct" trajectory becomes training data and teaches the model a wrong pattern — and in on-policy training, where you filter by your own reward, this poison compounds. Two defenses:

1. **Discard degenerate matches** when building training data: empty result sets, and trivial queries like bare `SELECT *`, are too easy to match by coincidence — don't trust them as "correct" for *training* (they're fine to count in eval if you want, but be aware).
2. **Use test-suite accuracy where you can.** Spider provides a test-suite evaluation that runs each query against *multiple* database instances (generated by perturbing the data), so a coincidental match on one instance gets caught. Use the official Spider evaluation script for your *reported* numbers; use your fast in-process comparator for the *training-data filter* and quick iteration.

> **Task for Claude Code:** *"Integrate the official Spider test-suite evaluation script as an alternative scoring path in `src/eval.py`, behind a `--official` flag, while keeping the fast in-process `execution_match` as the default for iteration."*

### 4.2 The tools

```python
# src/tools.py  (sketch)
def list_tables(db_id): ...                  # SELECT name FROM sqlite_master WHERE type='table'
def get_schema(db_id, tables=None): ...      # return CREATE TABLE statements (+ FKs)
def sample_values(db_id, table, column, k=5):# SELECT DISTINCT col FROM table LIMIT k
    ...
def execute_sql(db_id, sql): ...             # run_query with timeout; return rows or error text
def submit(sql): ...                         # marker that ends the episode; returns the sql
```

Expose these to the model as **tool/function definitions** (JSON schemas) via the chat-completions tool-calling API. The model picks a tool and arguments; your loop executes it and returns the result as a tool message.

### 4.3 The agent loop (the heart of the "agents" topic — understand every line)

```python
# src/agent.py  (model-agnostic ReAct loop)
from openai import OpenAI   # works for any OpenAI-compatible endpoint (incl. vLLM)

def run_episode(client, model, db_id, question, tool_defs, max_steps=8):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},   # explains tools + that it must call submit()
        {"role": "user", "content": f"Database: {db_id}\nQuestion: {question}"},
    ]
    for step in range(max_steps):
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=tool_defs, temperature=0.0,
        )
        msg = resp.choices[0].message
        messages.append(msg)                              # record the assistant turn (thoughts + tool call)

        if not msg.tool_calls:                            # model answered without a tool — nudge or stop
            continue

        for call in msg.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments)
            if name == "submit":
                return {"submitted_sql": args["sql"], "messages": messages, "steps": step + 1}
            result = dispatch_tool(name, db_id, args)     # calls into src/tools.py
            messages.append({                             # feed the OBSERVATION back
                "role": "tool", "tool_call_id": call.id, "content": str(result),
            })
    return {"submitted_sql": None, "messages": messages, "steps": max_steps}  # gave up
```

Things to get right (and reason about):

- **`temperature=0.0`** for the teacher and for eval (deterministic, reproducible). You'll *raise* temperature later when you want *diverse* student rollouts for rejection sampling — that's a deliberate change in a different script, not a change to eval.
- **`max_steps`** caps runaway loops. A model that never calls `submit` must terminate eventually. Track how often this happens — frequent timeouts is a signal (often value-grounding or schema confusion).
- **The full `messages` list IS the trajectory.** Save it verbatim; it's your training data in Phase 1.
- **System prompt** must clearly describe each tool and instruct the model to finish by calling `submit`. Spend time here; a vague system prompt produces garbage trajectories.

> **Task for Claude Code:** *"Implement `dispatch_tool(name, db_id, args)` that routes to the functions in `src/tools.py` and returns a string-safe result, truncating any result set to the first 30 rows so the context doesn't explode."* (Truncating observations matters — a `SELECT *` on a big table will blow your context window otherwise.)

### 4.4 Establish the baseline

Serve the **untrained** student (`Qwen/Qwen2.5-Coder-7B-Instruct`, or the 3B variant on free Colab) with vLLM, point the agent loop at it, run over Spider `dev.json`, and compute execution accuracy with the official script. Append the result to `results/eval_runs.jsonl` with a label like `{"model": "qwen2.5-coder-7b-instruct", "phase": "baseline", "ex": 0.xx}`.

```python
# notebooks/02_serve_vllm.ipynb — serve, then hit it
!python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --enable-auto-tool-choice --tool-call-parser hermes &
# then from the eval driver:
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
```

> vLLM's tool-calling flags and parser names depend on the model and vLLM version — check current vLLM docs for the right `--tool-call-parser` for Qwen. This is exactly the kind of version-sensitive detail to verify live rather than trust from any guide.

**Why Qwen2.5-Coder specifically (don't skip this reasoning):** the starting model dominates the final result. Teams doing this have found that initializing from a model *already fluent in code and SQL* consistently beats a general-purpose base, and a weak starting model often *never catches up* during fine-tuning. Starting from a general 3B chat model is the easiest way to make yourself conclude "distillation doesn't work" when really you just started from the wrong place.

### 4.5 Exit criteria for Phase 0

- [ ] Comparator returns sane results on a handful of hand-checked examples (including an intentionally-wrong query → `False`, and a reordered-but-equivalent query → `True`).
- [ ] Agent loop completes episodes, calls tools, and submits on a few questions you watch end-to-end.
- [ ] You have a baseline EX number for the untrained student, logged to `results/`.
- [ ] The whole eval run is cheap and repeatable with one command.

Realistic baseline expectation: a 7B coder model zero-shot in an agent loop on Spider dev lands somewhere in the broad 40–65% EX range depending on your prompt and tools. **The number doesn't matter; having a trustworthy number does.** Don't tune to chase it — you have three phases to improve it.

---

## 5. Phase 1 — Off-policy distillation (= fine-tuning)

**Goal:** make a strong teacher solve Spider training questions in *your* agent loop, keep only its successful trajectories, and QLoRA-fine-tune the student to imitate them. This is sequence-level distillation and ordinary SFT at the same time — feel the equivalence.

### 5.1 Generate teacher trajectories

Run `generate_trajectories.py`: the same `run_episode` loop, but `model` = your teacher endpoint, over `train_spider.json`. Start with a **subset** — 1,500–3,000 questions — not all 7k. You can always generate more; you can't un-spend the money.

```bash
python src/generate_trajectories.py \
  --split train --teacher deepseek-chat --limit 2000 \
  --out data/trajectories/teacher_raw.jsonl
```

Each line: the question, db_id, gold_sql, the full `messages` transcript, the submitted SQL, and step count.

**Cost reality:** these trajectories are short (a few thousand tokens each including tool round-trips). With a cheap teacher, 2,000 trajectories is single-digit to low-tens of dollars. This is the only real spend in the off-policy phase. Watch your provider's usage dashboard the first 50 calls to sanity-check per-trajectory cost before launching the full batch.

### 5.2 Filter for correctness (the most important step)

```bash
python src/filter.py \
  --in data/trajectories/teacher_raw.jsonl \
  --out data/trajectories/teacher_correct.jsonl
```

`filter.py` runs each submitted SQL through `execution_match` and keeps only:
- trajectories where the submitted SQL matches the gold result, **and**
- **not** a degenerate match (drop empty-result and trivial `SELECT *` matches — see §4.1).

Expect to keep maybe 60–85% of teacher trajectories depending on teacher strength. If your yield is very low, your tools/prompt are probably the problem, not the teacher — fix that before spending more on generation.

> **Why filtering is the whole game:** the Snowflake Arctic-Text2SQL team found that model-based correctness filtering was what turned noisy synthetic SQL data into a useful training signal — clean-and-correct examples are worth far more than raw volume. Your execution check *is* that filter, for free.

### 5.3 Format as training data (HF "messages" format)

TRL's trainers consume conversations in the standard `messages` format: a list of `{role, content}` turns, including the tool calls and tool results, ending with the model's `submit`. `format_sft.py` converts each *correct* trajectory into one training example in that shape, then writes `data/sft/train.jsonl`.

The model is trained to predict the **assistant turns** (its thoughts + tool calls + final submit) given the system/user/tool context. Crucially, you want the *whole successful trajectory* as the target, so the student learns the agentic behavior (explore → sample → query → submit), not just final-SQL generation.

> **Task for Claude Code:** *"Write `src/format_sft.py` that reads filtered trajectories and emits JSONL where each line is `{\"messages\": [...]}` in the format `SFTTrainer` expects, preserving tool-call and tool-result turns. Make sure the tokenized length distribution is reported so I can set `max_seq_length`."* Knowing your length distribution prevents the silent truncation that quietly corrupts training.

### 5.4 QLoRA SFT on Colab (Unsloth)

Upload `train.jsonl` (git push or Drive), open `01_sft_unsloth.ipynb`. The core:

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig          # TRL is patched by Unsloth
import torch

max_seq_length = 4096   # set from your length distribution (§5.3)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen2.5-Coder-7B-Instruct",
    max_seq_length = max_seq_length,
    dtype = None,                # auto: bf16 on Ampere+, fp16 on T4
    load_in_4bit = True,         # QLoRA
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 32,
    target_modules = ["q_proj","k_proj","v_proj","o_proj",
                      "gate_proj","up_proj","down_proj"],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",   # memory saver for long context
    random_state = 3407,
)

trainer = SFTTrainer(
    model = model, tokenizer = tokenizer,
    train_dataset = load_dataset("json", data_files="train.jsonl", split="train"),
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,       # effective batch size 8
        warmup_steps = 5,
        num_train_epochs = 2,                  # 1–3; watch for overfit
        learning_rate = 2e-4,
        logging_steps = 5,
        optim = "adamw_8bit",
        seed = 3407,
        output_dir = "outputs",
    ),
)
trainer.train()

# save the adapter (small) and/or a merged 16-bit model for vLLM
model.save_pretrained("qwen-sql-sft-lora")                     # adapter only
model.save_pretrained_merged("qwen-sql-sft-merged", tokenizer, # for vLLM serving
                             save_method="merged_16bit")
```

What to actually watch:
- **Training loss should drop smoothly.** If it's flat, LR is too low or data is malformed. If it explodes, LR too high.
- **2 epochs is a reasonable start.** More epochs on a few thousand examples overfits — the model memorizes trajectories and *task* accuracy stops improving even as loss keeps dropping. This is the trap from §1.2 made real.
- **Persist the output to Drive/Hub immediately.**

### 5.5 Re-evaluate and log

Serve `qwen-sql-sft-merged` with vLLM, run the **frozen** eval over Spider dev, log to `results/`. Compare to baseline.

**Exit criteria:** EX is meaningfully above baseline (even +5–15 points is a real, demonstrable win). If it *didn't* move or dropped, debug in this order: (1) is the data format correct and not truncated? (2) did you over-train (try 1 epoch)? (3) is your filter actually removing wrong trajectories? Do **not** proceed to Phase 2 on top of a broken Phase 1.

---

## 6. Phase 2 — On-policy distillation

**Goal:** improve the student by training it on *its own* trajectories rather than the teacher's. This is the conceptually richest phase and the one that most clearly separates "I fine-tuned a model" from "I understand distillation." Do 2A first; 2B is the stretch.

### 6.1 Why bother (recap, because this is the point your manager will probe)

Your Phase-1 student learned to imitate the teacher's *perfect* transcripts. But at test time it generates its *own*, sometimes-flawed intermediate states — states it never saw in training — so it doesn't know how to recover. On-policy training closes this train/inference gap by training on the student's own rollouts. The risk is error cascade (one bad tool call poisons the rest); short SQL trajectories keep that manageable, which is exactly why this task was chosen over long-horizon ones.

### 6.2 Phase 2A — Rejection-sampling fine-tuning (robust, do this first)

The simplest correct form of on-policy learning. The loop:

1. Take your **Phase-1 student**. Set a **higher temperature** (e.g. 0.7–1.0) so it produces *diverse* rollouts.
2. Have it attempt Spider **training** questions, **several samples each** (e.g. 4–8 rollouts per question).
3. Run every rollout through the execution check. **Keep the ones that got the right answer** (and aren't degenerate matches).
4. These verified-correct, *self-generated* trajectories become new training data.
5. QLoRA-fine-tune again on them (start from the Phase-1 adapter or from base — try both).
6. Re-evaluate on dev. Optionally iterate another round.

```bash
# Colab: 03_rejection_sampling.ipynb (conceptual flow)
# - serve phase-1 student with vLLM at temperature 0.8
# - for each train question: sample k=6 rollouts via the SAME agent loop
# - filter.py keeps correct, non-degenerate ones
# - dedupe near-identical trajectories
# - format_sft.py -> selfgen_train.jsonl
# - run the Unsloth SFT cell on selfgen_train.jsonl
```

Why this works and is "on-policy": the model learns from states *it* actually visits, reinforcing its own successful problem-solving paths. It's also *online-ish* if you iterate rounds — and online interaction tends to beat one static batch because the model keeps encountering and learning from fresh negative examples it generates as it improves.

Watch for:
- **Diversity collapse:** if temperature is too low, all k rollouts are identical and you gain nothing. Dedupe and check you're getting variety.
- **The false-positive poison (§4.1) matters most here** — you're filtering by your *own* reward, so a lenient comparator will feed the model coincidentally-"correct" garbage. Keep the degenerate-match guards on.
- **Coverage:** hard questions where *no* rollout succeeds contribute no training signal. That's expected; note which question types never yield (great Phase-3 analysis material).

**Exit criteria:** EX above your Phase-1 number. Even a few points is a legitimate, well-understood result and exactly the "off-policy → on-policy" delta you're trying to show.

### 6.3 Phase 2B — Token-level on-policy distillation with TRL GKD (the stretch)

The advanced form: instead of a binary keep/discard reward, an **open-weights teacher** grades *every token* the student generates, and the student trains to match the teacher's per-token distribution. This is the technique from the on-policy distillation literature, and TRL's `GKDTrainer` implements it.

**Requirements:** the teacher must be open-weights (you need its logits), so run a 14–32B coder model on Colab as the teacher. This is more memory and more complexity — only attempt it once 2A works.

```python
# notebooks/04_gkd.ipynb (core)
from trl.experimental.gkd import GKDConfig, GKDTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

student = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")  # or your phase-1 merged
teacher = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-14B-Instruct") # open weights → logits
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")

config = GKDConfig(
    output_dir = "gkd-out",
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 8,
    learning_rate = 5e-5,
    lmbda = 0.5,      # fraction of ON-POLICY (student-generated) data. 0=off-policy, 1=fully on-policy
    beta  = 0.5,      # divergence: 0=forward KL, 1=reverse KL, between=generalized JSD
    max_new_tokens = 512,
)
trainer = GKDTrainer(model=student, teacher_model=teacher,
                     args=config, train_dataset=ds, processing_class=tokenizer)
trainer.train()
```

**The demonstration that proves you understand the concept:** run this **twice**, changing only `lmbda`:
- `lmbda = 0.0` (with `seq_kd`) ≈ off-policy / sequence-level distillation — the teacher's token probabilities on fixed data.
- `lmbda = 1.0` = fully on-policy — the student generates, the teacher grades its own outputs.

Compare the two dev-set results. If on-policy wins (the literature says higher `lmbda` usually does), you've *empirically reproduced* the core finding of modern distillation with your own harness. That is a genuinely strong thing to walk into an internship having done — and a far better answer to "what's the difference between off-policy and on-policy distillation?" than anything you could recite.

Caveats, stated plainly:
- GKD **generates during training** (for the on-policy fraction), so it's slow and memory-hungry. Keep sequences short, batch small, and expect to babysit OOM.
- Holding teacher and student on the **same tokenizer** keeps this simple — using the Qwen2.5-Coder family for both avoids a cross-tokenizer rabbit hole.
- This is the optional crown, not the core. A complete project is Phases 0–2A plus a clean writeup. 2B is upside.

---

## 7. Phase 3 — Scale up and analyze

Optional, but this is where the resume-grade insight lives.

**Swap Spider → BIRD.** BIRD has bigger, messier, real-world schemas, an "external knowledge / evidence" hint per question, and a second metric — the **Valid Efficiency Score (VES)**, which rewards *correct queries that also run fast* (it scores by the ratio of gold execution time to your query's time). Two decisions to make explicitly and hold constant: whether you feed the evidence hints (changes difficulty a lot), and you should run the **whole pipeline** (0→1→2) again on BIRD rather than just evaluating, so the deltas are apples-to-apples.

> BIRD is genuinely harder; small-model EX on BIRD is much lower than on Spider. Don't be discouraged — the *delta* across your phases is the result, not the absolute number.

**Error taxonomy (`src/analyze.py`).** Categorize failures into:
- **schema-linking errors** — wrong table/column chosen,
- **value-grounding errors** — wrong literal (the thing `sample_values` is supposed to prevent — check if your agent actually used it),
- **logic errors** — wrong join, aggregation, or filter structure,
- **looping/no-submit** — ran out of steps.

Then look at *how the distribution shifts* from baseline → off-policy → on-policy. A great finding is something like "off-policy fixed most schema-linking errors but on-policy was what reduced value-grounding errors, because the student learned to actually call `sample_values`." That sentence is exactly the kind of mechanistic understanding that makes you look like you get it.

**On VES:** once you have correct queries, you can show whether your fine-tuning made queries not just correct but *efficient* — a nice secondary axis almost nobody bothers to analyze.

---

## 8. The traps, consolidated

Ranked by how much time they'll cost you if ignored:

1. **Not building eval first.** Without a trustworthy metric you're flying blind and a "successful" training run means nothing. Phase 0 is the gate. (§4)
2. **Wrong starting model.** A general-purpose small model often never catches up. Start from Qwen2.5-Coder-Instruct. (§4.4)
3. **Execution-accuracy false positives.** Coincidental matches (empty sets, single values, `SELECT *`) poison training data, worst in on-policy. Guard the comparator; use test-suite eval for reported numbers. (§4.1, §6.2)
4. **Skipping the correctness filter / under-valuing data quality.** A few hundred clean trajectories beat thousands of noisy ones. The filter is the whole game. (§5.2)
5. **Over-training.** Loss down ≠ accuracy up. Watch the *task metric*, not the loss. 1–3 epochs. (§1.2, §5.4)
6. **Capacity ceiling at tiny sizes.** Below ~7B (or ~3B as a stretch) the agent loops and scores near zero; you'll wrongly blame distillation. (§1.2)
7. **Silent truncation.** Trajectories longer than `max_seq_length` get cut, corrupting training. Check your length distribution. (§5.3)
8. **Inconsistent harness between phases.** Change the eval or tools and your deltas are garbage. Freeze after Phase 0. (§0)
9. **Losing Colab work to disconnects.** Persist adapters/trajectories to Drive/Hub the moment they exist. (§3.4)
10. **Over-generating teacher data.** Start with 1.5–3k trajectories; scale only if needed. Money you can't get back. (§5.1)

---

## 9. What "done" looks like

A complete, defensible project is:

- A **frozen eval harness** with execution accuracy (test-suite-backed) on Spider dev.
- A **results table / plot**: baseline → off-policy SFT → on-policy, showing a rising EX curve. This single artifact *is* the deliverable.
- A short **README writeup** that, for each of the three topics, states what you did, the number it moved, and *why* (the mechanism). Bonus: the GKD `lmbda` 0-vs-1 comparison and the error-taxonomy shift.

How to talk about it (since this is internship prep): lead with the pipeline and the curve, then be ready to go deep on the *distinctions* — off-policy vs on-policy, why filtering matters, why the agent's `sample_values` tool exists, why loss isn't the metric. The fact that you reproduced the off-policy→on-policy improvement with your own harness, on a task with a free objective reward, is the thing that signals you actually understand this rather than having followed a tutorial. That your manager named these exact three topics strongly suggests the team does small-model agent-distillation work — showing up having built this end to end means you've already touched their stack.

**Minimum viable version** (if time gets short): Phases 0, 1, and 2A on Spider, with the results curve and writeup. That alone hits all three topics with a measured result. Everything else (2B, BIRD, VES, taxonomy) is genuine upside, not a requirement.

---

## 10. Resources

- **Unsloth** — fast QLoRA fine-tuning, free Colab notebooks, docs: `unsloth.ai/docs` and `github.com/unslothai/unsloth`.
- **TRL** — `SFTTrainer` for Phase 1, `GKDTrainer` for Phase 2B (on-policy distillation), docs: `huggingface.co/docs/trl`. The GKD trainer page explains `lmbda`/`beta` directly.
- **Spider** — official site (Yale LILY group) and Hugging Face mirror; includes the **test-suite evaluation** scripts you should use for reported numbers.
- **BIRD** — official BIRD-bench site; defines EX and VES; for Phase 3.
- **Qwen2.5-Coder** — model cards on Hugging Face (`Qwen/Qwen2.5-Coder-7B-Instruct`, `-3B-`, `-1.5B-`).
- **vLLM** — OpenAI-compatible serving incl. tool calling; check current docs for the right `--tool-call-parser` for Qwen.
- **Concepts** — search the GKD paper ("On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes") and recent on-policy-distillation writeups for the theory behind Phase 2.

> Versions and product details (Unsloth/TRL/vLLM APIs, Claude Code install) move quickly. When a snippet here disagrees with current official docs, trust the docs — and lean on Claude Code to reconcile the difference against the version you actually installed.

---

*Build the eval first. Freeze the harness. Filter ruthlessly. Watch the metric, not the loss. Everything else is detail.*
