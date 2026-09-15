# LIMO — Less Is More for Reasoning

A faithful, minimal, CPU-only reproduction of the central thesis of
**"LIMO: Less Is More for Reasoning"**, Ye, Huang, Wu, Li, Liu, Xu, Song, Liu,
Wang, Wu, Liu — 2025, [arXiv:2502.03387](https://arxiv.org/abs/2502.03387).
Everything needed to read, run, and understand the paper's core claim lives in
this folder.

## Plain-English summary

LIMO challenges the assumption that eliciting strong reasoning requires massive
instruction-tuning datasets. Its finding: with a capable base model, a **few
hundred carefully curated, high-quality reasoning demonstrations** unlock
stronger reasoning than **orders of magnitude more** mediocre data. The quality
of the *reasoning process* shown in each example matters far more than the sheer
number of examples. LIMO calls this the **Less-Is-More Reasoning hypothesis**.

This folder reproduces that effect at tiny scale by fine-tuning the *same* small
model two ways, at an **identical training budget**, and comparing held-out
reasoning accuracy:

- **Curated (few, high-quality)** — a few hundred **step-by-step** solutions
  that expose every intermediate result.
- **Bulk (many, low-quality)** — an order of magnitude more **answer-only**
  "shortcut" solutions that give the right answer but demonstrate no reasoning.

Both sets contain only *correct* answers — the sole difference is the quality of
the demonstrated reasoning.

## What the demo shows

The task is chained single-digit addition **modulo 10** (e.g.
`7+7+4+3 → 1`). The reasoning quality is the independent variable:

| Quality | Example demonstration | What it teaches |
|---|---|---|
| High (curated) | `7+7+4+3=7+7=4;4+4=8;8+3=1#1` | a **reusable** `(a+b) mod 10` step, applied everywhere |
| Low (bulk) | `7+7+4+3=#1` | only the whole 4-input function, from answers alone |

The single-step skill in the high-quality trace is shared across every step and
every problem, so the model learns it from just a few hundred examples and
generalizes. The one-shot function `(Σ digits) mod 10` is hard to learn from
answers alone, so even 10× more low-quality data barely beats chance.

In the reference run (600 identical steps, batch 64 for both):

```
curated  ( 200 high-quality): 99.3% held-out accuracy   (fewer training tokens)
bulk     (2000 low-quality ): 10.3% held-out accuracy
→ the 10× smaller curated set wins by ~89 points
```

The curated model *shows its work* and is right; the bulk model jumps straight
to a (usually wrong) guess.

## Folder layout

```
LIMO/
├── limo.pdf                   # the paper
├── requirements.txt           # pinned deps (CPU PyTorch + numpy)
├── src/
│   ├── model.py               #   tiny decoder-only reasoner (TinyGPT)
│   └── task.py                #   the task + two demonstration qualities + tokenizer
├── data/
│   └── generate_data.py       #   materializes the curated & bulk datasets to JSON
├── demo/
│   └── run_demo.py            #   SFT both models at equal budget, compare accuracy
└── visualization/
    └── index.html             #   interactive: accuracy vs dataset-size bars + traces
```

## Setup

Requires Python 3.10+. Reuse the shared virtual environment, or create one:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## How to run

```bash
python data/generate_data.py    # (optional) writes the two datasets to data/*.json
python demo/run_demo.py          # SFT curated + bulk, evaluate held-out (~53s on CPU)
```

The demo writes `data/limo_run.json` (metrics + example generations) for the
visualization.

## Expected output

```
Held-out reasoning accuracy (unseen problems):
  curated  ( 200 high-quality, ~1.08M train tokens):  ~99%
  bulk     (2000 low-quality , ~0.42M train tokens):  ~10%
  → curated set is 0.10x the size but wins by ~89 points

Sample generations (curated model shows its work):
  7+7+4+3=7+7=4;4+4=8;8+3=1#1.   [OK]
Sample generations (bulk model jumps to an answer):
  7+7+4+3=#6.   [WRONG]
OK: fewer high-quality reasoning traces beat many low-quality ones.
```

## Explore the visualization

Open `visualization/index.html` in any browser (works offline via `file://`).
It plots the accuracy bars against the dataset-size bars — a striking visual of
a 10× smaller dataset winning by a wide margin — and shows the curated vs
low-quality traces plus real generations from both models. To load fresh data,
serve the folder:

```bash
python -m http.server 8000    # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper idea | Where | File |
|---|---|---|
| Less-Is-More Reasoning hypothesis | few high-quality > many low-quality | `demo/run_demo.py` |
| Quality of the reasoning chain | step-by-step vs answer-only demonstrations | `src/task.py` |
| Curated demonstration set | a few hundred high-quality traces | `build_dataset(..., "high")` |
| Supervised fine-tuning (SFT) | next-token training of the reasoner | `demo/run_demo.py` → `train()` |
| Held-out reasoning evaluation | accuracy on unseen problems | `demo/run_demo.py` → `evaluate()` |

## Honest scope notes

This is a *mechanism* reproduction, not a reproduction of LIMO's numbers. The
real LIMO fine-tunes a strong pretrained 32B model on 817 curated math
solutions; here the "reasoner" is a ~0.4M-parameter model trained from scratch
on a synthetic arithmetic task. Consequently we do **not** rely on a powerful
pretrained prior — instead we make the reasoning skill *reusable across steps* so
that a small curated set can teach it, which is the same underlying reason
high-quality reasoning traces are so sample-efficient in LIMO. The comparison
holds the model, step count, and batch size fixed, varying only the training
data, and reports the (smaller) token budget the curated set actually uses.
