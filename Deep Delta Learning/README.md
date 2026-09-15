# Deep Delta Learning

A minimal, self-contained reproduction of the core mechanism of Zhang, Liu,
Wang & Gu, *"Deep Delta Learning"* (2026), arXiv:**2601.00417**. Everything
needed to read, run, and understand the mechanism lives in this folder: the
paper PDF, a from-scratch PyTorch implementation, a runnable CPU demo, and an
interactive visualization.

> **Honesty note.** This is a 2026 paper. The demo reproduces the paper's most
> characteristic *documented* mechanism — the **delta rule** as a
> **read → compare → rewrite** operation, in both of its roles: (1) a
> fast-weight **associative memory** and (2) Deep Delta Learning's **depth-wise
> residual interface** with its local error-correction identity
> `e_post = (1 − β)·e_pre` (paper §2, Eq. 2.2–2.4) — at tiny CPU scale on
> synthetic data. It does **not** reproduce the GPT-2-scale pretraining on
> FineWeb-Edu, the expanded-state Transformer, or the reported downstream
> numbers. It isolates and demonstrates the delta rewrite itself.

## What the paper is about

Standard Transformer residual blocks update the residual stream additively,
`X_{l+1} = X_l + F_l(X_l)`, leaving reading, comparison, and replacement
implicit inside an unconstrained branch. **Deep Delta Learning (DDL)** makes
those operations explicit with a depth-wise **delta rule**:

```
readout  r = k_lᵀ X_l                                   # read the state
X_{l+1} = X_l + β_l k_l (v_l − k_lᵀ X_l)ᵀ               (Eq. 2.2)
```

along a learned unit direction `k_l`, toward a target `v_l`, with a gate
`β_l ∈ (0,2)`. Its signature property (Eq. 2.4) is that a single rewrite scales
the selected-readout error by exactly `(1 − β)`: β=0 is the identity, β=1 is an
exact overwrite, and 1<β<2 is an over-relaxed correction. The same rule, as a
matrix state `W`, is the classic **associative memory** `W ← W + β (v − W k) kᵀ`.

## What the demo shows

`demo/run_demo.py` runs two parts on CPU in about a second:

- **Part A — Associative memory.** Store key→value pairs in one pass and recall
  them (100% retrieval when lightly loaded). Rewriting a key **replaces** its
  value (recall matches the *new* value, not a superposition). A capacity curve
  shows graceful degradation as the number of pairs approaches the memory's
  dimension.
- **Part B — Depth-wise DDL.** Numerically verify `e_post = (1 − β)·e_pre` to
  machine precision across β, and show that stacking DDL layers drives a readout
  of the residual state to its target, contracting the error by `(1 − β)` per
  layer.

## Folder layout

```
Deep Delta Learning/
├── deep_delta_learning.pdf     # the paper
├── requirements.txt            # pinned deps (CPU PyTorch + numpy)
├── src/
│   ├── delta_memory.py         #     fast-weight associative memory (W ← W + β(v−Wk)kᵀ)
│   └── ddl.py                  # §2  depth-wise delta rewrite + DDLBlock, Eq. 2.2-2.4
├── data/
│   ├── generate_demo_data.py   # write a human-readable key/value sample
│   └── results.json            # metrics written by the demo (generated)
├── demo/
│   └── run_demo.py             # end-to-end CPU demo (memory + depth-wise DDL)
└── visualization/
    └── index.html              # interactive, offline: memory + error-correction
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
python demo/run_demo.py                 # main demo (~1s on CPU)
python data/generate_demo_data.py       # optional: inspect stored/recalled pairs
```

## Expected output

```
PART A  Delta-rule associative memory: store once, recall
  Retrieval accuracy: 100.0%   (mean-squared recall error 0.0042)
  Overwrite test: recall vs v_old cos=+0.035   recall vs v_new cos=+1.000  -> latest value recalled

PART B  Deep Delta Learning: depth-wise error correction
  Verify e_post = (1 - beta) * e_pre  (Eq. 2.4):
    beta=1.00  e_pre=3.5076  e_post=0.0000   # exact overwrite
  max deviation from the identity: 3.58e-07
```

Exact numbers vary slightly with the environment, but recall is ~100% when
lightly loaded, overwrites recall the latest value, and the error-correction
identity holds to machine precision. The demo exits non-zero otherwise.

## Explore the visualization

Open `visualization/index.html` in any browser (offline via `file://`). It shows
the read→compare→write pipeline and equations, the associative-memory recall /
overwrite results and capacity curve, the `e_post = (1−β)·e_pre` table, and the
depth-convergence chart — all from the real `data/results.json`. For live data:

```bash
python -m http.server 8000    # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper section | Concept | Where in code |
|---|---|---|
| §1 | Delta rule as fast-weight memory | `DeltaAssociativeMemory` in `src/delta_memory.py` |
| §2.1, Eq. 2.2 | Depth-wise delta rewrite `X + β k (v − kᵀX)ᵀ` | `ddl_step`, `DDLBlock` in `src/ddl.py` |
| §2.1, Eq. 2.3 | Exact overwrite at β=1 | `ddl_step` + Part B identity check |
| §2.1, Eq. 2.4 | Error update `e_post = (1−β)·e_pre` | `demo/run_demo.py` Part B |
| §2.2 | Expanded residual state (d_v channels) | `DDLBlock(d_val=...)` |
| §1 | Store/recall & overwrite behaviour | `demo/run_demo.py` Part A |
