# DeepSeek-V4 — The Muon Optimizer (+ mHC)

A minimal, self-contained, CPU-only reproduction of DeepSeek-V4's most
distinctive *training* ingredients: the **Muon optimizer** (orthogonalize the
momentum update via hybrid Newton–Schulz iterations) and, as a bonus, the
**Manifold-Constrained Hyper-Connections (mHC)** residual constraint.

- **Paper:** *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*
- **Authors:** DeepSeek-AI
- **Year:** 2026 · **arXiv:** 2606.19348

> **Scope & honesty.** DeepSeek-V4 is a frontier MoE model with heavily
> compressed attention and million-token context; it cannot be reproduced here.
> This demo reproduces two *documented* mechanisms from scratch at tiny CPU
> scale — the Muon optimizer (Section 2.4, Algorithm 1 & Eq. 28) and the mHC
> doubly-stochastic residual projection (Section 2.2) — and empirically shows
> Muon's convergence benefit and its Newton–Schulz orthogonalization at work.

## What the demo shows

- **Muon vs Adam.** Two identical MLPs (same init, same data) fit a
  teacher network. Muon reaches Adam's final loss in fewer steps.
- **Newton–Schulz orthogonalization.** We trace the singular values of a random
  matrix across the 10 hybrid Newton–Schulz iterations and watch them converge
  to 1.0 — the mechanism that turns the momentum into an orthogonal update.

Representative output (seed 0):

```
final MSE   Muon : 0.00723
final MSE   Adam : 0.00723
Muon reaches Adam's final loss in step 195  (1.5x fewer steps)
Newton–Schulz singular values: start range [0.007, 0.384] -> final [1.000, 1.001]
```

## The Muon update (Algorithm 1)

```
Gₜ = ∇L(Wₜ₋₁)                              # gradient
Mₜ = μ·Mₜ₋₁ + Gₜ                           # momentum buffer
O′ = HybridNewtonSchulz(μ·Mₜ + Gₜ)         # Nesterov + orthogonalize
O  = O′ · √max(n,m) · γ                     # rescale update RMS
Wₜ = Wₜ₋₁·(1 − ηλ) − η·O                    # weight decay + step
```

Hybrid Newton–Schulz (Eq. 28), applied after `M₀ = M / ‖M‖_F`:

```
Mₖ = a·Mₖ₋₁ + b·(Mₖ₋₁Mₖ₋₁ᵀ)Mₖ₋₁ + c·(Mₖ₋₁Mₖ₋₁ᵀ)²Mₖ₋₁
steps 1–8 : (a,b,c) = (3.4445, −4.7750, 2.0315)   # fast
steps 9–10: (a,b,c) = (2, −1.5, 0.5)              # stabilize
```

1-D parameters (biases, norms, embeddings) use Adam, as in the paper.

## Folder layout

```
DeepSeek-V4/
├── deepseek-v4.pdf           # the paper
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── muon.py               # Muon optimizer + hybrid Newton–Schulz
│   ├── mhc.py                # doubly-stochastic (Sinkhorn) residual projection
│   └── model.py              # tiny MLP under test
├── data/
│   ├── generate_data.py      # teacher-student regression dataset
│   ├── regression.json       # (generated)
│   └── muon_result.json      # (generated) loss curves + Newton–Schulz trace
├── demo/
│   └── run_demo.py           # Muon vs Adam + Newton–Schulz trace
└── visualization/
    └── index.html            # Muon algorithm, σ→1 chart, loss curves
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

Runs both optimizers and the Newton–Schulz trace in ~2s on CPU and writes
`data/muon_result.json` for the visualization.

## Expected output

The demo asserts Muon matches or beats Adam's final loss and that the
Newton–Schulz singular values converge into `[0.8, 1.2]` (they land at ~1.000),
then prints `OK: Muon converges (>=) as well as Adam; ...`.

## Explore the visualization

Open `visualization/index.html` in any browser (offline via `file://`): the
Muon algorithm, a chart of singular values funneling to 1.0 across the 10
Newton–Schulz iterations, and the measured Muon-vs-Adam loss curves. For live
data, serve the folder:

```bash
python -m http.server 8000   # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper section | Idea | File |
|---|---|---|
| §2.4, Alg. 1 | Muon update (momentum, Nesterov, rescale, decay) | `src/muon.py` → `Muon` |
| §2.4, Eq. 28 | Hybrid Newton–Schulz orthogonalization | `src/muon.py` → `newton_schulz` |
| §2.4 | Adam fallback for 1-D parameters | `src/muon.py` → `_adam_step` |
| §2.2 | mHC doubly-stochastic residual constraint | `src/mhc.py` → `doubly_stochastic` |
