# s1 — Simple Test-Time Scaling (Budget Forcing)

A faithful, minimal, self-contained reproduction of the signature idea of
Muennighoff et al., *"s1: Simple test-time scaling"* (2025), arXiv:2501.19393.
Everything needed to read, run, and understand the idea lives in **this folder**:
the paper PDF, a from-scratch implementation, a runnable CPU demo, and an
interactive visualization.

## Plain-English summary

s1 shows you can control how long a model "thinks" — and thereby its accuracy —
with a dead-simple decoding trick called **budget forcing**:

- to make the model think **more**, suppress its end-of-thinking token and append
  the word **"Wait"**, which makes it continue reasoning;
- to make it think **less**, force the end-of-thinking token to stop it early.

Paired with a **tiny curated SFT set** (~1K examples) to teach the reasoning
format, budget forcing yields a clean test-time-scaling curve: more forced
thinking → higher accuracy.

## What this demo does (and how it simplifies the paper)

The task is multi-step modular arithmetic (start from a value in `0..9`, apply a
chain of operations, answer mod 10). Two learned pieces stand in for an LLM:

- `step_net` — an MLP that executes one operation at a time;
- `stop_logits` — a learned **stop head**, SFT-trained on a **1000-example
  curated set** that is *skewed toward short chains* (Poisson-shaped lengths). It
  therefore learns to stop thinking around step ~3 and **under-thinks** on longer
  problems.

**Budget forcing** then overrides the stop head at decode time: whenever the model
wants to stop before a minimum number of steps, we append **"Wait"** and force
another reasoning step. Sweeping that minimum reproduces the scaling curve.

**Honesty note:** the model is small and the task tiny, but the mechanism is the
paper's exactly — a curated SFT'd stop behaviour, overridden by "Wait" to extend
test-time compute.

## What the demo shows

```
Model's natural stopping point: ~3 steps (it under-thinks on longer chains)

  forced min-thinking  0 | accuracy  42.5% | avg steps  2.6 | avg 'Wait's  0.0
  forced min-thinking  4 | accuracy  53.5% | avg steps  3.2 | avg 'Wait's  0.6
  forced min-thinking  6 | accuracy  75.0% | avg steps  4.1 | avg 'Wait's  1.5
  forced min-thinking  8 | accuracy 100.0% | avg steps  4.5 | avg 'Wait's  1.9

Example: 0 -2 +2 *3 -1 +4 -1 -3 +4 = ?   true=3
  natural : <think> start=0 -2->8 +2->0 *3->0 </think> answer=0  WRONG (3 steps)
  forced  : <think> start=0 -2->8 +2->0 *3->0 Wait -1->9 Wait +4->3 Wait -1->2
                    Wait -3->9 Wait +4->3 </think> answer=3  OK (8 steps, 5 'Wait's)
```

Natural decoding stops after ~3 steps and is wrong on long chains; each "Wait"
buys another reasoning step, and accuracy climbs from 42.5% to 100%. Runs in ~4s
on CPU. Results are written to `data/s1_results.json` for the viz.

## Folder layout

```
s1 - Simple Test-Time Scaling/
├── s1_-_simple_test-time_scaling.pdf   # the paper
├── requirements.txt                     # pinned deps (CPU PyTorch)
├── src/
│   ├── task.py                          # task + curated (short-skewed) SFT sampler
│   └── reasoner.py                      # step model + stop head + budget forcing
├── data/
│   └── generate_data.py                 # writes curated/test length histograms
├── demo/
│   └── run_demo.py                      # SFT + budget-forcing sweep
└── visualization/
    └── index.html                       # budget-forcing mechanism + scaling curve
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
python demo/run_demo.py         # SFT + budget-forcing sweep, ~4s on CPU
```

Deterministic (`--seed`). Try `--max-len`, `--curated-size`, `--sft-iters`.

## Explore the visualization

Open `visualization/index.html` in any browser (offline via `file://`). It shows
the budget-forcing mechanism (the "Wait" injection that extends a real trace) and
the accuracy-vs-forced-thinking curve. Ships with baked-in data; served over HTTP
it reloads live results:

```bash
python -m http.server 8000   # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper idea | Where it lives | File |
|---|---|---|
| Tiny curated SFT (~1K) | short-skewed curated set | `src/task.py` → `make_problems(skew=True)` |
| Learned when-to-stop | SFT'd stop head | `src/reasoner.py` → `stop_logits`, `demo/run_demo.py` → `sft_stop_head` |
| Budget forcing: "Wait" to think more | override early stop | `src/reasoner.py` → `decode(min_think=...)` |
| Budget forcing: end token to stop | thinking cap | `src/reasoner.py` → `decode(max_think=...)` |
| Test-time scaling curve | accuracy vs forced thinking | `demo/run_demo.py`, viz |
