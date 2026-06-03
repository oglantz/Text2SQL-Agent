# text2sql-distill

A self-contained learning project: build a tool-using SQL agent, harvest a strong
model's behavior, and compress it into a small open model — measuring execution
accuracy at every step.

See [`docs/guide.v2.0.md`](docs/guide.v2.0.md) for the full end-to-end guide.

## The pipeline

```
strong model acts as an AGENT  →  harvest its successful trajectories
        →  FINE-TUNE a small model on them (off-policy DISTILLATION)
        →  student generates its own trajectories, train on those (on-policy DISTILLATION)
        →  measure execution accuracy after each step
```

The agent loop, the tools, and the eval comparator stay **frozen** across every
phase. The only thing that changes is the model weights.

## Repo layout

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

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
cp .env.example .env            # then fill in your teacher API key
```

## Status

Scaffold only — implementation tracked by phase in the guide. Build the eval
first, freeze the harness, filter ruthlessly, watch the metric not the loss.
