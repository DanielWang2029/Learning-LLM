# GLM-5 — DeepSeek Sparse Attention (DSA)

A minimal, self-contained, CPU-only reproduction of the **core efficiency
mechanism** of GLM-5: **DeepSeek Sparse Attention (DSA)**. Everything needed to
read, run, and understand the idea lives in this folder — the paper PDF, a
from-scratch implementation, a runnable demo with data, and an interactive
visualization.

- **Paper:** *GLM-5: from Vibe Coding to Agentic Engineering*
- **Authors:** GLM-5 Team, Zhipu AI & Tsinghua University
- **Year:** 2026 · **arXiv:** 2602.15763

> **Scope & honesty.** GLM-5 is a 744B-parameter (40B active) frontier MoE model
> trained on 28.5T tokens; it cannot be reproduced here. This demo reproduces
> the paper's most characteristic *documented* architectural mechanism — DSA's
> lightning indexer + top-k sparse attention (Section 2.1.1) — from scratch at
> tiny CPU scale, and empirically demonstrates the effect DSA is built for:
> lossless attention while scoring far fewer keys. Agentic RL, MoE scaling, and
> the async RL infrastructure are described in the paper but out of scope here.

## What the demo shows

DSA's philosophy is to replace dense `O(L²)` attention with a **content-based,
fine-grained selection**: a cheap *lightning indexer* scores how relevant each
key is to each query, and only the **top-k** keys receive real softmax
attention. The demo follows the paper's DSA "warm-up" recipe (train *only* the
indexer while the base attention stays frozen) and measures:

- **attention mass recall** — how much of the true dense-attention mass the
  top-k keys capture (random ≈ `k/L`; a trained indexer ≈ 1.0),
- **output fidelity** — cosine similarity between the sparse and full outputs,
- **compute saved** — `1 − k/L`, since only `k ≪ L` keys are scored per query.

Representative output (seed 0):

```
attention mass recall :  10.0% (untrained) ->  93.4% (trained indexer)
output cosine vs full : 0.9997
keys scored per query : 8 of 64  (compute saved: 88%)
```

So with 88% of the keys skipped, the sparse output is essentially identical to
full attention — the "lossless by construction" property the paper claims.

## Folder layout

```
GLM-5/
├── glm-5.pdf                 # the paper
├── requirements.txt          # pinned CPU dependencies
├── src/
│   ├── __init__.py
│   └── dsa.py                # lightning indexer, full & top-k sparse attention
├── data/
│   ├── generate_data.py      # tiny synthetic sequence with sparse attention
│   ├── sample_sequence.json  # (generated) input tokens
│   └── dsa_result.json       # (generated) metrics + arrays for the viz
├── demo/
│   └── run_demo.py           # trains the indexer, prints evidence, writes JSON
└── visualization/
    └── index.html            # interactive DSA diagram + measured results
```

## Setup

Requires Python 3.10+. From inside this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

CPU-only PyTorch — no GPU needed.

## How to run

```bash
python demo/run_demo.py
```

It generates the sample sequence if missing, trains the indexer (~400 steps),
prints the metrics above in well under a minute, and writes
`data/dsa_result.json` for the visualization.

## Expected output

The demo asserts `mass recall > 0.9` and `output cosine > 0.9` and prints
`OK: DSA top-k approximates full attention while scoring far fewer keys.` on
success. Mass recall jumps from ~10% (untrained) to ~93% (trained), and the
sparse output matches full attention at cosine ≈ 0.9997.

## Explore the visualization

Open `visualization/index.html` in any browser (works offline via `file://`).
It shows the DSA pipeline, the indexer/attention equations, the measured stats,
a dense-attention heatmap with the **selected top-k keys outlined** (they land
on the high-mass cells), a mass-recall bar chart, and a linear-vs-quadratic cost
curve. For live data instead of the baked-in sample, serve the folder:

```bash
python -m http.server 8000   # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper section | Idea | File |
|---|---|---|
| §2.1.1 | Lightning indexer `I[t,s] = Σ_h w_h·ReLU(qᴵ·kᴵ)` | `src/dsa.py` → `LightningIndexer` |
| §2.1.1 | Top-k selection per query | `src/dsa.py` → `dsa_attention` |
| §2.1.1 | Sparse softmax attention over k keys | `src/dsa.py` → `dsa_attention` |
| §2.1.1 | Dense reference (what DSA approximates) | `src/dsa.py` → `full_attention` |
| §2.1.1 | Indexer warm-up (train indexer, freeze base) | `demo/run_demo.py` |
| Abstract | Cut long-context cost, keep fidelity | mass-recall + cosine metrics |
