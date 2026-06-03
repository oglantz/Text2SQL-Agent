# LoRA / QLoRA Hyperparameters — Detailed Reference

A companion to §1.2 of the project guide. This explains, in depth, each hyperparameter you set when fine-tuning with LoRA/QLoRA: what it *is* (with the math where the math actually clarifies), what it controls, how to choose it, how it fails, and how the knobs interact. The values referenced throughout match the training config in §5.4 of the main guide (`r=32`, `lora_alpha=32`, all seven projections, `lora_dropout=0`, `learning_rate=2e-4`).

---

## Read this first: which knobs actually matter

Brutally honest prioritization, because tuning the wrong ones wastes your iteration budget:

| Hyperparameter | How much it moves your result | What to do |
|---|---|---|
| **learning rate** | **A lot.** The single most impactful knob. | The one you actually tune. Watch the loss curve. |
| **`r` (rank)** | Moderate. Matters if you're under/overfitting. | Set 32; adjust only if diagnostics say to. |
| **`lora_alpha`** | Low *if* you follow the convention. | Set `= r` (or `2r`) and forget. |
| **`target_modules`** | Low. The standard set is just correct. | Use all seven projections. Never touch. |
| **`lora_dropout`** | Low for this dataset. | Set 0. Raise only if overfitting persists. |

The takeaway: **learning rate is where your iteration cycles should go.** The other four have well-established defaults that are right for this project; changing them is mostly motion, not progress. The rest of this doc explains *why* each default is the default, so you can recognize the rare cases where deviating is justified.

---

## The shared mechanism (understand this once, the rest follows)

Every LoRA hyperparameter is a knob on one idea. A pretrained weight matrix `W₀` has shape `d × k` (e.g. 4096 × 4096). A full fine-tune would learn an update `ΔW` of the *same* `d × k` shape and add it: `W₀ + ΔW`. That's `d·k` trainable numbers per matrix — millions, times every matrix in the model. Unaffordable.

LoRA's bet: the *useful* update `ΔW` has low **intrinsic rank** — it can be well-approximated by multiplying two skinny matrices. So instead of learning the full `ΔW`, you learn:

```
ΔW = B · A
        where  B has shape d × r   (tall and thin)
               A has shape r × k   (short and wide)
               r ≪ min(d, k)       (the "rank" — the bottleneck)
```

The model's forward pass becomes (with the scaling factor explained under `lora_alpha`):

```
h = W₀·x  +  (α / r) · B · A · x
    └─frozen─┘  └────── trainable adapter ──────┘
```

Concrete parameter count, `d = k = 4096`, `r = 16`:
- Full update: `4096 × 4096` = **16.8M** params.
- LoRA: `4096×16 + 16×4096` = `16 × 8192` = **131K** params.
- ~**128× fewer** trainable parameters, for that one matrix.

**Initialization detail that matters:** `A` is random (small Gaussian), `B` is initialized to **zero**. So at step 0, `B·A = 0`, which means `ΔW = 0` and the model behaves *exactly* like the untouched pretrained model. Training then grows the adapter from "no effect" — you never start by disrupting the base weights. This is why LoRA is stable and why you can use aggressive learning rates (see below).

Everything below is a knob on this structure: `r` is the bottleneck width, `lora_alpha` scales the adapter's contribution, `target_modules` chooses *which* `W₀`s get an adapter, `lora_dropout` regularizes the adapter, and `learning_rate` sets how fast `A` and `B` move.

---

## 1. `r` — the rank (adapter capacity)

### What it is
The inner dimension of the `B·A` decomposition — the width of the bottleneck. It is the number of independent "directions" the adapter is allowed to use to modify each weight matrix.

### What it controls
The **capacity** (expressiveness) of the adapter. Higher `r` means the adapter can represent a richer, more complex transformation — closer to what a full fine-tune could do. Lower `r` forces the adapter to compress the task adaptation into fewer directions.

It also directly sets the trainable parameter count and thus memory/compute: params scale linearly with `r`. Doubling `r` doubles the adapter size.

### The tradeoff
- **Too low:** the adapter can't capture the task even with enough data and training — *underfitting*. Symptom: training loss plateaus high, task metric stays near baseline no matter how long you train.
- **Too high:** more capacity than the task/data needs — *overfitting* risk on a small dataset, plus wasted memory and slower training. Symptom: training loss drops nicely but execution accuracy on dev stalls or *degrades* (the model memorized your trajectories instead of learning to write SQL — the exact trap from §1.2).

The non-obvious empirical fact from the original LoRA work: useful intrinsic rank is often *surprisingly small* — even `r = 1–4` works for some tasks. For task adaptation like text-to-SQL, the returns flatten quickly above the 16–32 range. Going to 128 rarely helps on a few-thousand-example dataset and often hurts.

### How to choose
Start at **32** (the guide's value). Then let diagnostics drive any change:
- Underfitting (loss won't drop, metric flat) → raise `r` (32 → 64), *or* suspect data/format first.
- Overfitting (loss drops, dev metric stalls/drops) → lower `r` (32 → 16) and/or add data and/or cut epochs.

For this project, `r = 32` is a deliberately safe middle. If you run the optional 7B-vs-bigger ablation, holding `r` fixed while you change *model size* keeps that comparison clean.

### Interactions
- With `lora_alpha`: the adapter's *effective strength* is `alpha/r`, so changing `r` while holding `alpha` fixed silently changes that strength (see §2). If you raise `r` and keep `alpha = r`, you're keeping strength at 1.0, which is usually what you want.
- With learning rate: more capacity can need slightly more care on LR, but the `alpha/r` normalization mostly handles this.

---

## 2. `lora_alpha` — the scaling factor

### What it is
A scalar that scales the adapter's contribution to the forward pass. The adapter output is multiplied by `alpha / r` before being added to the frozen base output:

```
h = W₀·x + (alpha / r) · B·A·x
```

**The thing to internalize: what matters is the *ratio* `alpha / r`, not `alpha` alone.** `alpha = 32, r = 32` gives scaling `1.0`. `alpha = 64, r = 32` gives scaling `2.0` — the adapter's effect is doubled.

### Why this knob exists (the part that's genuinely useful)
It **decouples the adapter's magnitude from its rank.** Without the `1/r` normalization, changing `r` would change the typical magnitude of `B·A` (more rank → more accumulated magnitude), which would force you to re-tune the learning rate every time you adjusted `r`. The `alpha/r` scaling normalizes the contribution so that, roughly, you can change `r` *without* re-tuning everything else. `alpha` then lets you dial the overall adapter influence independently.

### The tradeoff
A higher effective scaling (`alpha/r`) amplifies the adapter's signal — it behaves *somewhat like* a higher learning rate on the adapter path. Too high and you can get instability or the adapter overpowering the base knowledge; too low and the adapter barely influences the output and learning is sluggish.

But within the normal conventions, this is a low-risk knob — which is why it's near the bottom of the priority table.

### How to choose
Follow the convention and stop thinking about it:
- `alpha = r` → scaling 1.0 (conservative, stable; the guide's choice: `alpha=32, r=32`).
- `alpha = 2r` → scaling 2.0 (a very common default; "alpha twice the rank," slightly more adapter influence).

Both are fine. Only deviate if you have a specific reason (e.g. you've concluded the adapter is underpowered and you'd rather bump `alpha` than LR).

### A footnote worth knowing (so a config flag doesn't confuse you)
The linear `alpha/r` scaling is theoretically suboptimal at *high* ranks; **rank-stabilized LoRA (rsLoRA)** instead scales by `alpha/√r`, which is the `use_rslora=True` flag you'll see in Unsloth/PEFT. It mainly matters when you push `r` large (64–256+). At your `r = 16–32`, standard linear scaling is the right call and `use_rslora` should stay `False`.

### Interactions
- With `r`: inseparable — the effective knob is the ratio. Change them together intentionally.
- With learning rate: raising `alpha/r` and raising LR push in similar directions (stronger/faster adapter updates). Don't crank both at once or you'll lose track of which caused instability.

---

## 3. `target_modules` — which matrices get an adapter

### What it is
The list of weight matrices inside each transformer layer that receive a LoRA adapter. A Llama/Qwen-style layer has these big linear projections:

**Attention block:**
- `q_proj` — builds the query vectors
- `k_proj` — builds the key vectors
- `v_proj` — builds the value vectors
- `o_proj` — projects the attention output back to model width

**MLP / feed-forward block (gated, as in Llama/Qwen):**
- `gate_proj` — the gating projection
- `up_proj` — expands to the larger intermediate width
- `down_proj` — contracts back to model width

The guide targets **all seven**.

### What it controls
*Coverage* — how much of the network the adapter is allowed to modify. Adapting only attention leaves the MLP frozen; adapting everything lets the fine-tune touch both where the model *attends* and where it *computes/stores* transformations.

### The tradeoff and the history (so you know why "all seven" is right)
- The **original LoRA paper** adapted only attention matrices, and often found `q_proj` + `v_proj` alone were enough — this kept adapters minimal.
- The **QLoRA paper and subsequent practice** found that adapting **all linear layers** (attention + MLP) consistently gives better results, and because adapters are tiny relative to the base model, the extra cost is negligible. This is now the standard recommendation.

So adapting all seven is the well-established default: better coverage, trivial added cost. More target modules = more adapter params, but we're talking about going from ~0.5% to ~1% of model parameters — nothing that strains your A100.

### How to choose
Use the standard seven. This is the most "set and forget" knob in the list. Two notes:
- Many libraries accept `target_modules="all-linear"` to auto-target every linear layer — equivalent in spirit, convenient if a model uses different names.
- You generally do **not** adapt embeddings, the LM head, or layernorms. The exception is when you're teaching genuinely new tokens/vocabulary (you'd then also train embeddings) — not relevant to this project.

### Interactions
- With `r`: every targeted module gets its own rank-`r` adapter, so total adapter params = (number of targeted matrices) × (per-matrix LoRA params). Targeting all seven and using `r=32` is jointly cheap; don't worry about the product here.

---

## 4. `lora_dropout` — regularization on the adapter

### What it is
Dropout applied to the LoRA path during training. Dropout randomly zeroes a fraction `p` of activations on each forward pass, forcing the network not to over-rely on any specific units — a standard regularization technique. It is **off at inference**; it only perturbs training. `lora_dropout` applies this specifically to the adapter's input, with `p` typically in `0` to `~0.1`.

### What it controls
Overfitting resistance for the adapter. Higher dropout = stronger regularization = the adapter generalizes a bit better but learns a bit more slowly and noisily.

### The tradeoff
- **`0`:** no regularization from this knob; fastest, cleanest signal. Risk of overfitting if you have little data and train many epochs. Also the *optimized* path in Unsloth — there's a faster kernel when dropout is 0, so you get a small speed win for free.
- **`0.05–0.1`:** meaningful regularization; useful when you see overfitting you can't otherwise kill. Slightly slower convergence.

### How to choose
Start at **0** (the guide's value), for three reasons specific to this project: the dataset is only a few thousand trajectories, you train just 1–2 epochs (already strong overfitting control), and your *primary* overfitting levers are epoch count and data quality, not dropout. Dropout is a secondary lever here.

Raise to `0.05–0.1` only if, after reducing epochs and verifying data quality, you *still* see the overfitting signature (train loss down, dev EX stalling/dropping).

### Interactions
- With epochs and dataset size: dropout, low epoch count, and more data are three substitutable ways to fight overfitting. With few epochs and a clean filtered dataset, you rarely need dropout.
- With the Unsloth fast path: keeping it at exactly `0` unlocks the optimized kernel — a minor but free reason to leave it there unless you need otherwise.

---

## 5. `learning_rate` — the step size (the one you actually tune)

### What it is
How far the optimizer moves the trainable weights (here, the adapter matrices `A` and `B`) in response to each batch's gradient. The fundamental knob of gradient descent.

### What it controls
The speed and stability of learning. It is **the most impactful hyperparameter in the whole list** — get it badly wrong and nothing else you set matters.

### Why LoRA tolerates a *higher* LR than full fine-tuning (important, non-obvious)
In a **full** fine-tune, you're nudging pretrained weights that are already in a carefully-balanced good state. Large steps risk *catastrophic forgetting* — destabilizing knowledge the model already has — so full FT uses small LRs, typically `1e-5` to `5e-5`.

In **LoRA**, the base weights are frozen and untouchable, and the adapter starts at zero-effect (`B=0`). You're not risking the base knowledge at all — you're growing a small, isolated module from scratch. That lets you take *bigger* steps to learn the task faster without wrecking anything. Hence the standard LoRA/QLoRA band of `1e-4` to `3e-4` — roughly an order of magnitude higher than full FT.

### Typical values and the guide's choice
`1e-4` to `2e-4` is the standard starting band. The guide uses **`2e-4`**, a common, slightly-aggressive-but-safe default for QLoRA task adaptation.

### Failure modes (and how to read the loss curve)
Watch the loss in the **first ~20 steps** — it tells you almost immediately:
- **Too high:** loss spikes, oscillates wildly, or diverges; sample outputs turn to garbage. → Drop to `1e-4`, then `5e-5`.
- **Too low:** loss barely moves, training crawls, model underfits in your epoch budget. → Raise toward `3e-4`.
- **Healthy:** loss descends smoothly and steadily, with normal small noise.

This fast feedback is exactly why LR is the knob to spend iteration cycles on — a bad value is cheap to detect and fix.

### The schedule (warmup + decay)
LR is rarely held constant. Two standard components, both handled by the trainer config in §5.4:
- **Warmup** (`warmup_steps=5`): ramp the LR up from ~0 over the first few steps. Early gradients are noisy and the adapter is fresh; ramping avoids an early destabilizing jolt.
- **Decay** (cosine or linear, via the trainer's scheduler): gradually lower the LR over training so late updates are gentle and the model settles. The default scheduler handles this.

So `2e-4` is the *peak* LR; the actual rate ramps up to it, then decays.

### Interactions
- With `lora_alpha`/`r`: a higher effective adapter scaling (`alpha/r`) amplifies updates much like a higher LR. Tune one or the other, not both simultaneously, or you can't attribute instability.
- With batch size (`per_device_train_batch_size` × `gradient_accumulation_steps`): larger effective batch → less noisy gradient → can often support a slightly higher LR (the rough "linear scaling rule"). If you change your effective batch size a lot, revisit LR.
- With epochs: LR and epoch count jointly determine total movement. A high LR for many epochs is a fast route to overfitting on a small dataset.

---

## Related setting you'll see in the same config (not strictly a LoRA hyperparameter)

**`load_in_4bit=True` / quantization (NF4).** This is the "Q" in QLoRA: the frozen base model is stored in 4-bit NormalFloat precision to slash memory, while the adapters stay in higher precision (bf16) so the training signal isn't degraded. It's a memory/quality setting rather than a tuning knob — you leave it `True` because it's what makes a 7B fit comfortably (and is why §1.2's memory math collapses from ~112 GB to ~12 GB). One caution echoed in the main guide: some newer model families quantize poorly to 4-bit; on a model where 4-bit is stable (Qwen2.5-Coder is fine), this is a non-issue.

---

## Recommended starting config for this project (and why)

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 32,                       # safe-middle capacity; raise only if underfitting
    lora_alpha = 32,              # = r → effective scaling 1.0, conservative/stable
    target_modules = ["q_proj","k_proj","v_proj","o_proj",
                      "gate_proj","up_proj","down_proj"],  # all linear: standard best practice
    lora_dropout = 0,             # tiny dataset + few epochs; also Unsloth's fast path
    use_rslora = False,           # standard linear scaling is right at r=32
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)
# ... in SFTConfig:
#   learning_rate = 2e-4         # THE knob to watch; adjust off the loss curve
#   num_train_epochs = 2         # primary overfitting control
#   warmup_steps = 5             # gentle LR ramp
```

The logic in one line: **four of these are principled defaults you set and leave; `learning_rate` (with `num_train_epochs` close behind) is where your actual experimentation lives.**

---

## Troubleshooting by symptom

| Symptom | Most likely cause | First fix → then |
|---|---|---|
| Loss spikes / diverges in first steps | LR too high | LR `2e-4 → 1e-4 → 5e-5`; confirm warmup is on |
| Loss barely moves | LR too low, or malformed data | Check data format/truncation first; then LR up toward `3e-4` |
| Loss drops, but dev EX stalls/drops | Overfitting | Cut epochs (2→1) → lower `r` (32→16) → add/clean data → add `lora_dropout=0.05` |
| Loss plateaus high, EX near baseline | Underfitting | Verify data quality → raise `r` (32→64) → check enough epochs |
| OOM during training | Sequence length / batch too big for card | Lower `max_seq_length` or batch; ensure gradient checkpointing on (you have the A100, so this is rare) |

Remember the meta-rule from the main guide: **the loss going down is not the goal — execution accuracy going up is.** Every knob here is ultimately judged against the frozen eval harness, not the training curve.
