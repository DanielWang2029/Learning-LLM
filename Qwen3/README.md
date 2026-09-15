# Qwen3 — Hybrid Thinking & the Thinking Budget

A faithful, minimal, self-contained reproduction of the signature idea of the
**Qwen3 Technical Report** (Alibaba, 2025), arXiv:2505.09388. Everything needed
to read, run, and understand the idea lives in **this folder**: the paper PDF, a
from-scratch implementation, a runnable CPU demo, and an interactive
visualization.

## Plain-English summary

Qwen3's headline contribution is a **single model with two modes**:

- **thinking mode** — the model reasons step by step inside `<think>…</think>`
  before answering;
- **non-thinking mode** — the model answers directly, for speed.

Crucially, Qwen3 exposes a **thinking budget**: the caller controls how much
step-by-step computation the model may spend. More budget → more accurate on
hard, multi-step problems (at the cost of more tokens); less budget → faster.

## What this demo does (and how it simplifies the paper)

We can't run a real LLM on CPU, so we reproduce the *structure* on a task whose
answer genuinely needs sequential computation: start from a value in `0..9` and
apply a chain of modular-arithmetic operations (`+2`, `*3`, `-1`, …). The answer
is the final running value mod 10.

Two heads, trained on the same task, stand in for the model's two modes:

- **thinking path** (`StepReasoner`) — an MLP that learns one operation at a time
  (`value, op -> value`). At inference it *executes* the chain one step per
  "thinking token", up to the caller's **thinking budget**.
- **non-thinking path** (`DirectAnswerer`) — an MLP that must map the whole
  problem to the answer in a single shot, with no intermediate work.

**Honesty note:** these are small MLPs, not a shared transformer, and the task is
tiny — but the demo faithfully exercises Qwen3's core mechanism: *the same task,
answered with or without step-by-step thinking, under a controllable budget.*

## What the demo shows

```
  thinking budget  0 | accuracy  10.0% | avg tokens  0.0
  thinking budget  2 | accuracy  30.2% | avg tokens  1.9
  thinking budget  4 | accuracy  55.8% | avg tokens  3.3
  thinking budget  6 | accuracy  80.2% | avg tokens  4.2
  thinking budget  8 | accuracy 100.0% | avg tokens  4.5

Non-thinking (one-shot) accuracy: 21.5%
Thinking @ full budget (8) accuracy: 100.0%

Example (length 8): 7 +1 *3 +3 -3 +4 +2 +3 +1 = ?   true=4
  non-thinking -> 5  WRONG
  thinking     -> <think> start=7 +1->8 *3->4 +3->7 ... +1->4 </think> answer=4  OK
```

Accuracy climbs monotonically with the thinking budget and saturates at 100%
once the budget covers the whole chain — far above the non-thinking baseline.
Runs in ~5s on CPU. Results are written to `data/qwen3_results.json` for the viz.

## Folder layout

```
Qwen3/
├── qwen3.pdf                 # the paper
├── requirements.txt          # pinned deps (CPU PyTorch)
├── src/
│   ├── task.py               # multi-step modular-arithmetic task + <think> traces
│   └── reasoner.py           # StepReasoner (thinking) + DirectAnswerer (non-thinking)
├── data/
│   └── generate_data.py      # writes a few readable example problems
├── demo/
│   └── run_demo.py           # trains both paths, sweeps the thinking budget
└── visualization/
    └── index.html            # thinking/non-thinking toggle + accuracy-vs-budget curve
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
python demo/run_demo.py         # train + budget sweep, ~5s on CPU
```

Deterministic (`--seed`). Try `--max-len`, `--step-train-iters`, `--lr`.

## Explore the visualization

Open `visualization/index.html` in any browser (offline via `file://`). It shows
the thinking vs non-thinking toggle, a transparent `<think>` step trace, and the
accuracy-vs-thinking-budget curve with the tokens-used cost. Ships with baked-in
data; served over HTTP it reloads live results:

```bash
python -m http.server 8000   # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper idea | Where it lives | File |
|---|---|---|
| Single model, thinking mode | step-by-step execution | `src/reasoner.py` → `StepReasoner.run` |
| Single model, non-thinking mode | one-shot answer | `src/reasoner.py` → `DirectAnswerer.answer` |
| `<think>…</think>` reasoning trace | running-value trace | `src/task.py` → `render_think` |
| Controllable thinking budget | `budget` argument to the run loop | `demo/run_demo.py` |
| More budget → higher accuracy | accuracy-vs-budget sweep | `demo/run_demo.py`, viz |
