# LoRA: Low-Rank Adaptation of Large Language Models

A faithful, minimal, fully self-contained reproduction of Hu et al.,
*"LoRA: Low-Rank Adaptation of Large Language Models"* (2021),
arXiv:[2106.09685](https://arxiv.org/abs/2106.09685).

LoRA adapts a large **pretrained** model to a new task **without touching its
weights**. Instead of fine-tuning a weight matrix `W0`, it freezes `W0` and
learns a tiny low-rank update `ΔW = B·A`:

```
h = W0·x + b  +  (alpha / r) · (B · A) · x
```

with `A ∈ ℝ^{r×in}`, `B ∈ ℝ^{out×r}` and rank `r ≪ min(in, out)`. Only `A` and
`B` train, so the number of trainable parameters drops by ~two orders of
magnitude. `B` starts at zero, so training begins exactly at the pretrained
model; once trained, `ΔW` can be **merged** into `W0` for zero extra inference
cost, or **swapped** out to recover the original model.

To keep everything CPU-reproducible, the "pretrained model" is a small MLP and
the tasks are two related synthetic classification problems that share hidden
structure — so the *difference* between them has low intrinsic rank, exactly the
regime LoRA is built for (paper Section 7.2).

```
LoRA/
├── lora.pdf                 # the paper itself
├── requirements.txt         # pinned CPU dependencies (torch, numpy)
├── src/                     # the method, from scratch
│   ├── lora.py              #   §4  LoRALinear: W0 (frozen) + (alpha/r)·B·A ; merge/unmerge
│   ├── model.py             #   the tiny MLP backbone we adapt
│   └── data.py              #   two related toy tasks (shared low-rank structure)
├── data/                    # results JSON written by the demo (generated)
├── demo/run_demo.py         # pretrain -> full FT vs LoRA -> merge/swap, with evidence
└── visualization/index.html # ΔW=BA diagram, trainable-vs-frozen params, accuracy bars
```

## 1. Set up the environment

Requires Python 3.10+. Reuse the shared virtual environment:

```bash
source "/workspace/Attention Is All You Need/.venv/bin/activate"
```

Or build a fresh one:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## 2. Run the demo

```bash
python demo/run_demo.py
```

- **Stage 1 — Pretrain.** Train a small MLP on a base task and freeze it (`W0`).
- **Stage 2a — Full fine-tuning.** Train *all* weights on a new task (baseline).
- **Stage 2b — LoRA.** Freeze `W0`, inject `LoRALinear` adapters, and train only
  the rank-`r` factors `B, A` on the new task.
- **Stage 3 — Merge / unmerge / swap.** Fold `ΔW` into `W0` (output unchanged),
  undo it exactly, and swap the adapter off to recover the un-adapted model.

## 3. Expected output

Seeded and reproducible. A representative run (≈17 s on CPU):

```
full fine-tuning : 619,781 params -> 85.3%
LoRA (r=2)      : 6,218 params -> 81.3%
parameter saving : 99.7x fewer trainable params
accuracy gap     : +3.9 points

accuracy (LoRA active, unmerged) : 81.3%
accuracy (ΔW merged into W0)     : 81.3%   -> identical: True
accuracy (adapter swapped OFF)   : 9.4%   (back to the un-adapted model)

OK: LoRA matched full fine-tuning with ~100x fewer trainable params, ...
```

The three things that prove LoRA works, all asserted by the demo:

1. LoRA trains **~100x fewer** parameters than full fine-tuning,
2. it reaches accuracy **within a few points** of full fine-tuning on the new
   task, and
3. `merge`/`unmerge` are **bit-exact** (`ΔW` folds into `W0` with no change in
   output) and the adapter can be **swapped off** to restore the base model.

## 4. Explore the visualization

Open `visualization/index.html` in any browser (offline via `file://`). It
shows the low-rank decomposition `ΔW = B·A`, a trainable-vs-frozen parameter
breakdown, and accuracy bars comparing the pretrained model, full fine-tuning,
and LoRA — all baked in from your demo run. Serve the folder to load live data:

```bash
python -m http.server 8000   # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper section | Component | File |
|---|---|---|
| §4, Eq. (3) | `h = W0·x + (alpha/r)·B·A·x` | `src/lora.py` → `LoRALinear.forward` |
| §4.1 | `A ~ N(0,·)`, `B = 0` init, `alpha/r` scaling | `src/lora.py` → `LoRALinear.__init__` |
| §3 | Merge `ΔW` into `W0` (no inference latency) | `src/lora.py` → `merge` / `unmerge` |
| §1, §5 | Adapt frozen pretrained model to a new task | `demo/run_demo.py` |
| §7.2 | Low intrinsic rank of the update | `src/data.py` (shared low-rank tasks) |
