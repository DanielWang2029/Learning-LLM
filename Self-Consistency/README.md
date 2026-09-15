# Self-Consistency

A faithful, minimal, self-contained reproduction of the core idea of
Wang et al., *"Self-Consistency Improves Chain of Thought Reasoning in Language
Models"* (2022), arXiv:2203.11171. Everything needed to read, run, and
understand the idea lives in **this folder**: the paper PDF, a from-scratch
implementation, a runnable CPU demo, and an interactive visualization.

## Plain-English summary

Chain-of-thought prompting makes a model *show its reasoning*, but a single
reasoning chain can go wrong. **Self-consistency** replaces greedy decoding with
a simple idea: sample **many** diverse reasoning paths for the same question,
then keep the final answer that **most of them agree on** (a majority vote).
Because a correct answer can usually be reached by *several different* valid
lines of reasoning, those paths reinforce each other, while one-off mistakes are
scattered and get out-voted.

Why does voting beat greedy decoding? Greedy commits to the single
most-probable *path*. But the most-probable *answer* — obtained by
marginalizing over all the paths that reach it — can be different. Self-consistency
approximates "argmax over answers" instead of "argmax over paths".

## What this demo does (and how it simplifies the paper)

We cannot run a sampling LLM on CPU, so we use a **transparent programmatic
stochastic reasoner** (explicitly one of the paper's evaluation styles) on a
task that genuinely admits many reasoning paths: **adding several numbers**,
which is valid in any order.

- Every ordering of the additions is a distinct, correct reasoning path.
- On a fraction of "trap" problems there is also a tempting **shortcut** path
  (a systematic mistake — "skip the smallest addend") that has the *single
  highest* path-probability. Greedy decoding follows it and is **wrong**.
- Sampling (`temperature > 0`) spreads probability across the many correct
  orderings; their majority answer out-votes the shortcut.

`temperature` plays exactly the role it does in the paper: `T → 0` collapses onto
the single most-likely (trap) path; `T > 0` explores diverse correct paths.

**Honesty note:** the reasoner is a controlled stand-in, not a neural network.
It is engineered so that `argmax_path ≠ argmax_answer` — the precise condition
under which self-consistency helps — so the mechanism is visible and
deterministic. The takeaway matches the paper: majority-voting over sampled
chains beats a single greedy chain, and accuracy climbs with more samples.

## What the demo shows

```
============ ACCURACY: greedy vs self-consistency ============
  single greedy chain            :  57.2%
  majority vote over  1 samples  :  73.2%
  majority vote over  3 samples  :  80.0%
  majority vote over  5 samples  :  85.2%
  majority vote over  9 samples  :  85.2%
  majority vote over 15 samples  :  87.2%
  majority vote over 25 samples  :  89.5%
  majority vote over 41 samples  :  93.0%

  Example  19 + 78 + 22 = 119
    greedy path (lure) → 100  (WRONG)
    sampled votes: {119: 25, 100: 15, 129: 1}
    majority vote → 119  (correct)
```

Accuracy rises monotonically with the number of sampled paths, well above the
greedy baseline. Runs in a fraction of a second.

## Folder layout

```
Self-Consistency/
├── self-consistency.pdf         # the paper
├── requirements.txt             # (demo is pure stdlib; numpy optional)
├── src/
│   ├── reasoner.py              # stochastic chain-of-thought reasoner + task
│   └── self_consistency.py      # majority_vote over sampled answers
├── data/
│   └── generate_data.py         # writes a tiny human-readable sample
├── demo/
│   └── run_demo.py              # greedy vs self-consistency, accuracy-vs-samples
└── visualization/
    └── index.html               # interactive, colorful, offline
```

## Setup

Requires Python 3.10+. The demo needs nothing beyond the standard library, so
any Python works; you can also reuse the shared environment:

```bash
source "../Attention Is All You Need/.venv/bin/activate"
```

## How to run

```bash
python data/generate_data.py    # (optional) write data/sample_problems.json
python demo/run_demo.py         # greedy vs self-consistency comparison
```

Deterministic (`--seed`). Try `--temperature`, `--max-samples`, `--trap-frac`.

## Explore the visualization

Open `visualization/index.html` in any browser (offline via `file://`). It shows
the sample-paths → majority-vote pipeline, the accuracy-vs-#samples curve with
the greedy baseline, and the per-answer vote tally with real sampled traces.
Ships with baked-in data; served over HTTP it reloads live results:

```bash
python -m http.server 8000   # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper idea | Where it lives | File |
|---|---|---|
| Diverse reasoning paths for one question | every addition ordering | `src/reasoner.py` → `paths_for` |
| Sampling with temperature | softmax over path logits | `src/reasoner.py` → `sample_answer` |
| Greedy decoding baseline | argmax path | `src/reasoner.py` → `greedy_answer` |
| Marginalize by majority vote | mode of sampled answers | `src/self_consistency.py` → `majority_vote` |
| Accuracy vs number of sampled paths | the reported curve | `demo/run_demo.py` |
