# Gemini 2.5 — Thinking & Test-Time Compute Scaling

An honest, minimal, self-contained reproduction of the **documented headline** of
the **Gemini 2.5** report (Google, 2025), arXiv:2507.06261: a *thinking* model
whose accuracy scales with the amount of compute it is allowed to spend at test
time. Everything needed to read, run, and understand the idea lives in **this
folder**: the paper PDF, a from-scratch implementation, a runnable CPU demo, and
an interactive visualization.

## Honesty note (closed model)

Gemini 2.5 is a **closed** model: there are no released weights or training code.
What the report *does* document clearly is the behaviour we reproduce here — a
model that "thinks" before answering and gets more accurate when given more
test-time compute (a "thinking budget"), with accuracy eventually saturating. We
reproduce that **mechanism**, not the model, using the standard, well-documented
technique for test-time compute scaling: **parallel sampled reasoning paths
aggregated by majority vote** (self-consistency). This is the most characteristic
documented technique, implemented at small scale.

## Plain-English summary

A single reasoning attempt can slip on a hard, multi-step problem. If you let the
model **think several times** and then answer with the result it reaches most
often, the independent slips cancel out and accuracy climbs. Spend more test-time
compute (more reasoning paths / thinking tokens) → higher accuracy, until it
saturates near 100%.

## What this demo does

The task is 6-step modular arithmetic (start from a value in `0..9`, apply a chain
of operations, answer mod 10). A small MLP (`StepReasoner`) learns one operation
at a time. At test time each reasoning path is decoded **stochastically** (a
stand-in for an imperfect thinker), so different paths occasionally slip on
different steps. We then sweep the **test-time compute budget** = number of
sampled reasoning paths, aggregate by majority vote, and plot accuracy.

## What the demo shows

```
  paths(compute)   1 | thinking tokens    6 | accuracy  45.3%
  paths(compute)   2 | thinking tokens   12 | accuracy  44.0%
  paths(compute)   4 | thinking tokens   24 | accuracy  69.3%
  paths(compute)   8 | thinking tokens   48 | accuracy  75.3%
  paths(compute)  16 | thinking tokens   96 | accuracy  96.0%
  paths(compute)  32 | thinking tokens  192 | accuracy 100.0%
  paths(compute)  64 | thinking tokens  384 | accuracy 100.0%
  paths(compute) 128 | thinking tokens  768 | accuracy 100.0%

Example: 4 +1 +3 -3 *2 -3 *2 = ?  true=4
  single path (compute=1):   4
  majority of 128 paths:      4
  vote tally over answers 0..9: [23, 0, 10, 0, 66, 0, 9, 0, 20, 0]
```

Accuracy rises from ~45% (one path) to 100% (enough paths), then saturates — the
canonical test-time-compute-scaling curve. Runs in ~18s on CPU. Results are
written to `data/gemini25_results.json` for the viz.

## Folder layout

```
Gemini 2.5/
├── gemini_25.pdf             # the paper
├── requirements.txt          # pinned deps (CPU PyTorch)
├── src/
│   ├── task.py               # multi-step modular-arithmetic task
│   └── reasoner.py           # StepReasoner + self-consistency (majority vote)
├── data/
│   └── generate_data.py      # writes a few readable example problems
├── demo/
│   └── run_demo.py           # trains the thinker, sweeps the compute budget
└── visualization/
    └── index.html            # test-time scaling curve + vote ballot + notes
```

## Setup

Requires Python 3.10+. Reuse the shared virtual environment:

```bash
source "../Attention Is All You Need/.venv/bin/activate"
```

Or create a fresh one:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## How to run

```bash
python data/generate_data.py    # (optional) write data/sample_dataset.json
python demo/run_demo.py         # train + compute-budget sweep, ~18s on CPU
```

Deterministic (`--seed`). Try `--noise`, `--length`, `--budgets 1 4 16 64`.

## About Gemini 2.5's other headline claims

The report also emphasizes a very long context window (up to ~1M tokens) and
native **multimodality** (text, images, audio, video). Those are out of scope for
a CPU reproduction, but the visualization summarizes them alongside the test-time
scaling curve we do reproduce.

## Explore the visualization

Open `visualization/index.html` in any browser (offline via `file://`). It shows
the test-time scaling curve, an interactive majority-vote ballot for a real
problem, and notes on Gemini 2.5's long-context/multimodal claims. Ships with
baked-in data; served over HTTP it reloads live results:

```bash
python -m http.server 8000   # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Documented idea | Where it lives | File |
|---|---|---|
| Thinking model (step-by-step reasoning) | learned per-step reasoner | `src/reasoner.py` → `StepReasoner` |
| Test-time compute = more thinking | multiple sampled paths | `src/reasoner.py` → `self_consistency` |
| Majority vote over paths | vote tally + argmax | `src/reasoner.py` → `self_consistency` |
| Accuracy scales then saturates | compute-budget sweep | `demo/run_demo.py`, viz |
