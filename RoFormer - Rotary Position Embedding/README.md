# RoFormer — Rotary Position Embedding (RoPE)

A faithful, minimal, self-contained reproduction of **Rotary Position Embedding**
from Su et al., *"RoFormer: Enhanced Transformer with Rotary Position Embedding"*
(2021, [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)). Everything needed
to read, run and understand RoPE lives in this folder: the paper PDF, a
from-scratch implementation, a runnable demo, and an interactive visualization.

## The core idea, in plain English

Attention has no built-in notion of order, so we must inject position. RoPE does
it by **rotating** each query and key vector by an angle proportional to its
position. Split a vector into 2-D pairs; pair `i` is rotated by `m · θ_i` at
position `m`, with frequencies `θ_i = base^(−2i/d)` (paper Eq. 15).

The elegant consequence (paper §3.4.3): the attention score between a query at
position `m` and a key at position `n` becomes a function of the **relative**
distance `m − n` only:

```
⟨R(m) q, R(n) k⟩ = Σ_i |q_i||k_i| cos((m − n)·θ_i + φ_i) = f(m − n)
```

Because the rule is the same at every position, RoPE also **extrapolates** to
sequences longer than those seen in training.

## What the demo shows

`demo/run_demo.py` demonstrates two things:

1. **The relative-position property (numerically).** For fixed random `q, k` it
   builds the matrix `S[m,n] = ⟨R(m)q, R(n)k⟩` and shows the value is constant
   along each diagonal `m − n = const` — the max spread found along any diagonal
   is ~`5e-7` (floating-point noise), confirming the score depends only on `m−n`.

2. **Learning a relative task + extrapolation.** It trains a tiny one-layer
   attention model to "predict the token `k` positions earlier", comparing
   **RoPE** against a model with **no positional encoding**, then evaluates at
   sequence lengths *longer than training*. RoPE reaches ~100% and stays high
   when extrapolating; the no-position model sits near chance and degrades.

Results are written to `data/rope_results.json` for the visualization.

## Folder layout

```
RoFormer - Rotary Position Embedding/
├── roformer_-_rotary_position_embedding.pdf   # the paper
├── requirements.txt              # CPU PyTorch + numpy
├── src/
│   ├── rope.py                   # RoPE from scratch: frequencies, rotation, dot-product (§3)
│   └── model.py                  # tiny 1-layer attention LM: RoPE vs. no-position
├── data/
│   └── generate_data.py          # the "recall token k positions earlier" task
├── demo/
│   └── run_demo.py               # relative-position check + extrapolation experiment
└── visualization/
    └── index.html                # rotation demo + relative-property heatmap + extrapolation bars
```

## Setup

```bash
source "../Attention Is All You Need/.venv/bin/activate"
# or standalone:
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## How to run

```bash
python data/generate_data.py     # writes a small sample of the task to data/
python demo/run_demo.py          # both experiments; ~5s on CPU
```

## Expected output

```
Max spread along any diagonal: ~5e-07   (score depends only on m - n)

   seq len |  RoPE acc |  No-pos acc | note
        16 |    100.0% |       24.2% | train len
        24 |     99.8% |       21.3% | extrapolation
        32 |     99.5% |       19.3% | extrapolation
        48 |     98.9% |       16.7% | extrapolation
```

## Explore the visualization

Open `visualization/index.html` in any browser (offline via `file://`):

- **Rotation demo** — drag the position sliders and watch the angle between the
  rotated query and key depend only on `m − n`.
- **Relative-property heatmap** — the real `S[m,n]` matrix; hover to highlight a
  diagonal and see its values are (essentially) identical.
- **Extrapolation bars** — RoPE vs. no-position accuracy across lengths, with the
  training length marked.

Serve over HTTP to load live JSON: `python -m http.server 8000`.

## Code ↔ paper map

| Paper section | Idea | File |
|---|---|---|
| §3.2, Eq. 15 | Rotation frequencies `θ_i = base^(−2i/d)` | `src/rope.py` (`rope_frequencies`) |
| §3.2 | Rotate Q/K by position (`R(m)x`) | `src/rope.py` (`apply_rope`) |
| §3.4.3 | Score depends only on `m − n` | `demo/run_demo.py` (part a) |
| §3.3 | RoPE inside attention; extrapolation | `src/model.py`, `demo/run_demo.py` (part b) |
