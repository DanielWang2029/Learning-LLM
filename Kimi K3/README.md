# Kimi K3 — Kimi Delta Attention (KDA)

A minimal, self-contained, CPU-only reproduction of Kimi K3's core sequence
mixer: **Kimi Delta Attention (KDA)**, a *gated delta-rule linear attention*.
Everything needed to read, run, and understand it is in this folder.

- **Paper:** *Kimi K3: Open Frontier Intelligence*
- **Authors:** Kimi Team
- **Year:** 2026 · **arXiv:** 2607.24653

> **Scope & honesty.** Kimi K3 is a multi-trillion-parameter MoE with a 1M-token
> context and a layerwise KDA/Gated-MLA hybrid; it cannot be reproduced here.
> This demo reproduces KDA's *documented* recurrent mechanism (Section 2.1.1,
> Eq. 1) from scratch at tiny CPU scale and shows the two properties it exists
> for: reliable long-range memory and linear-time / constant-memory scaling.
> The chunkwise-parallel kernel, Gated MLA layers, and dual-axis scaling are
> described in the paper but out of scope here.

## What the demo shows

KDA keeps one **fixed-size** recurrent state `S ∈ R^{dk×dv}` and updates it with
a channel-wise forget gate plus the delta rule (Eq. 1):

```
Sₜ = (I − βₜ kₜ kₜᵀ)·Diag(αₜ)·Sₜ₋₁ + βₜ kₜ vₜᵀ        õₜ = Sₜᵀ qₜ
```

which rearranges into a decay → read → write-correction → output recurrence.
The demo:

- trains a tiny KDA model on an **associative-recall** task (store many
  key→value pairs, then retrieve one by its key), and
- times the layer at growing sequence lengths to show **linear** scaling (the
  state size is constant — there is no L×L attention matrix).

Representative output (seed 0):

```
final recall accuracy : 99.5%  (chance = 6.2%)
time / token (L=32)   : 223.88 us
time / token (L=256)  : 221.02 us   (flat => linear)
```

Flat time-per-token across a 8× length increase is the linear-time signature.

## Folder layout

```
Kimi K3/
├── kimi_k3.pdf               # the paper
├── requirements.txt
├── src/
│   ├── __init__.py
│   └── kda.py                # KDA layer (gated delta rule) + tiny KDAModel
├── data/
│   ├── generate_data.py      # associative-recall (MQAR-style) task generator
│   ├── recall_examples.json  # (generated) human-readable samples
│   └── kda_result.json       # (generated) accuracy + timing for the viz
├── demo/
│   └── run_demo.py           # trains recall, times linear scaling, prints evidence
└── visualization/
    └── index.html            # KDA recurrence diagram + accuracy & cost charts
```

## Setup

Requires Python 3.10+. From inside this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## How to run

```bash
python demo/run_demo.py
```

Trains for 400 steps and runs the timing sweep in ~35s on CPU, writing
`data/kda_result.json` for the visualization.

## Expected output

The demo asserts `recall accuracy > 0.9` (it reaches ~100%) and prints
`OK: KDA learns long-range key→value recall with a fixed-size linear-time
state.` The per-token forward time stays roughly constant as the sequence
length grows, confirming linear scaling.

## Explore the visualization

Open `visualization/index.html` in any browser (offline via `file://`): the KDA
recurrence and its decay/read/write/output steps, the measured recall-accuracy
curve, the measured (linear) forward-time sweep, and the asymptotic linear-vs-
quadratic op-count comparison. For live data, serve the folder:

```bash
python -m http.server 8000   # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper section | Idea | File |
|---|---|---|
| §2.1.1, Eq. 1 | Gated delta-rule state update | `src/kda.py` → `KDA.forward` |
| §2.1.1 | Channel-wise forget gate α, write strength β | `src/kda.py` → `w_alpha`, `w_beta` |
| §2.1.1, Eq. 2 | ShortConv + Swish + L2Norm on q/k/v | `src/kda.py` → `_short_conv`, `forward` |
| §2.1.1 | Fixed-size state ⇒ linear time | recurrence loop + timing sweep |
| §2.1 | KDA used as the model's mixer | `src/kda.py` → `KDAModel` |
