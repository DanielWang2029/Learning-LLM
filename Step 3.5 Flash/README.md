# Step 3.5 Flash — Sparse Mixture-of-Experts

A minimal, self-contained reproduction of the **efficiency idea** behind
StepFun Team, *"Step 3.5 Flash: Open Frontier-Level Intelligence with 11B
Active Parameters"* (2026), arXiv:**2602.10604**. Everything needed to read,
run, and understand the mechanism lives in this folder: the paper PDF, a
from-scratch PyTorch implementation, a runnable CPU demo, and an interactive
visualization.

> **Honesty note.** This is a 2026 paper. The demo reproduces the paper's most
> characteristic *documented* mechanism — a **sparse Mixture-of-Experts** with
> **top-k routing**, where total parameter capacity is large but only a few
> experts are *active* per token (paper Abstract, §2.2 "Sparse MoE Backbone")
> — at tiny CPU scale. It does **not** reproduce the 196B/11B model, the
> interleaved sliding-window/full attention, Multi-Token Prediction (MTP-3),
> the RL post-training, or any of the reported benchmark numbers. It isolates
> and demonstrates the total-vs-active-parameter trade-off and expert routing.

## What the paper is about

Step 3.5 Flash pairs a **196B-parameter foundation** (high total capacity) with
only **~11B active parameters** per token (cheap inference). The mechanism that
makes this possible is a **sparse MoE**: each layer holds many expert MLPs, and
a lightweight router sends each token to just the top-k of them. Total capacity
grows with the number of experts, but per-token compute grows only with `k`.

```
logits = x · W_r                               # score every expert
top-k  = argtop_k(softmax(logits))             # activate only k of N experts
y      = Σ_{i ∈ top-k}  softmax(logits)_i · Expert_i(x)
```

## What the demo shows

`demo/run_demo.py` trains three models on a **topic-conditioned symbol mapping**
task (each of 64 topics applies its own random permutation to 64 shared
symbols — 4096 input→output mappings). Because the symbol embeddings are shared
across topics, solving the task needs topic-dependent computation.

- **MoE** — 64 experts, top-1 routing. High total capacity, few active params.
- **dense-small** — a single MLP matched to the MoE's *active* compute.
- **dense-big** — a single MLP matched to the MoE's *total* parameters
  (all active every token).

The MoE matches the dense-big model's accuracy while activating only a small
fraction of its parameters, and it dramatically beats the active-matched
dense-small model — which lacks the capacity to hold all the topics at once.
The demo also measures that the router **specializes**: each topic's tokens are
routed far more concentratedly than uniform.

## Folder layout

```
Step 3.5 Flash/
├── step_35_flash.pdf           # the paper
├── requirements.txt            # pinned deps (CPU PyTorch + numpy)
├── src/
│   ├── moe.py                  # §2.2  SparseMoE, MoE/dense classifiers, param counters
│   └── task.py                 #        topic-conditioned symbol-mapping task
├── data/
│   ├── generate_demo_data.py   # write a human-readable sample of the task
│   └── results.json            # metrics + routing matrix written by the demo (generated)
├── demo/
│   └── run_demo.py             # end-to-end CPU demo (train + compare + routing)
└── visualization/
    └── index.html              # interactive, offline: total-vs-active, routing heatmap
```

## Setup

Requires Python 3.10+. Reuse the shared virtual environment shipped with the
repository:

```bash
source "../Attention Is All You Need/.venv/bin/activate"
```

Or create a fresh one:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## How to run

```bash
python demo/run_demo.py                 # main demo (~20s on CPU)
python data/generate_demo_data.py       # optional: inspect the task
```

## Expected output

```
Parameter budgets:
  MoE          total=214,240   active/token= 15,664   (7.3% active, top-1/64 experts)
  dense-small  total= 12,528   (matched to MoE ACTIVE compute)
  dense-big    total=208,080   (matched to MoE TOTAL params, all active)

Accuracy on held-out (symbol, topic) pairs:
  MoE (top-1/64)    99.9%   active/token=15,664
  dense-small      13.7%   active/token=12,528
  dense-big       100.0%   active/token=208,080

Routing is specialized (not uniform):
  busiest expert per topic handles 27.7% of its tokens (18x the 1.6% uniform baseline).
```

Exact numbers vary slightly with the environment, but the MoE reliably reaches
>90% while (a) beating the active-matched dense model by a wide margin and
(b) matching the fully-dense total-matched model at a fraction of the active
compute. The demo exits non-zero if this does not hold.

## Explore the visualization

Open `visualization/index.html` in any browser (offline via `file://`). It
shows the total-vs-active parameter split (with an expert grid you can hover),
the sparse-routing pipeline and equations, the accuracy/compute comparison, and
the 64×64 topic→expert routing heatmap — all from the real `data/results.json`.
For live data, serve the folder:

```bash
python -m http.server 8000    # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper section | Concept | Where in code |
|---|---|---|
| §2.2 | Sparse MoE backbone (many experts) | `SparseMoE`, `Expert` in `src/moe.py` |
| §2.2 | Top-k router + weighted combine | `SparseMoE.forward` |
| Abstract | Total vs active parameters (196B / 11B) | `count_params`, `active_params` |
| §2.1 | Design philosophy: capacity ≠ per-token compute | `MoEClassifier` vs `DenseClassifier` comparison |
| §2.3 | Routing behaviour / expert use | `routing_matrix` in `demo/run_demo.py` |
