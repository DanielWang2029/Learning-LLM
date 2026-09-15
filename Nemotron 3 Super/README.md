# Nemotron 3 Super — Hybrid Mamba-Attention

A minimal, self-contained, CPU-only reproduction of Nemotron 3 Super's
distinctive architecture: a **hybrid stack** that interleaves linear-time
**Mamba-2 SSM** layers with a few full **attention** layers.

- **Paper:** *Nemotron 3 Super: Hybrid Mamba-Transformer MoE*
- **Authors:** NVIDIA
- **Year:** 2026 · **arXiv:** 2604.12374

> **Scope & honesty.** Nemotron 3 Super is a 120B-total / 12B-active MoE trained
> in NVFP4 with LatentMoE; that cannot be reproduced here. This demo reproduces
> the paper's *documented core architecture* — a Mamba-2-style selective SSM
> layer and a hybrid stack alternating SSM with attention (Section 2 / Figure 2)
> — from scratch at tiny CPU scale. MoE routing and NVFP4 are omitted for size.

## What the demo shows

- **The hybrid learns a long-range task.** A small stack of mostly linear SSM
  layers plus one attention layer must recall a token from position 0 after a
  long span of distractors — it reaches 100% accuracy.
- **Linear vs quadratic cost.** We time one SSM layer against one attention layer
  as the sequence grows: the SSM's time grows *linearly* while attention's grows
  *super-linearly*, so attention's cost relative to the SSM keeps climbing.

Representative output (seed 0):

```
Hybrid stack: Mamba-SSM → Mamba-SSM → Attention
final recall accuracy : 100.0%  (chance = 10.0%)
     L |   SSM (ms) |  Attn (ms) |   ratio
    32 |       1.38 |       0.18 |   0.13x
   256 |      10.37 |       5.14 |   0.50x
```

SSM time ×7.5 while length ×8 (linear); attention time ×28 (quadratic). The
attention/SSM ratio climbs from 0.13× to 0.50× as length grows.

> Note: this naive Python SSM recurrence carries loop overhead, so it is slower
> in absolute wall-clock at these tiny sizes. The reproduced property is the
> **growth rate** (linear vs quadratic); real Mamba uses a parallel-scan kernel
> and is fast in wall-clock as well.

## Selective SSM recurrence (Mamba-2)

```
aₜ = exp(−Δₜ · exp(A))        # input-dependent decay in (0,1)
Hₜ = aₜ · Hₜ₋₁ + xₜ · Bₜ      # write the token into a fixed-size state
yₜ = ⟨Hₜ, Cₜ⟩ + D · xₜ        # read the state out
```

Δ, B, C are all functions of the current token (the "selective" mechanism), and
the state H has a fixed size, giving O(L) time and O(1) memory per step.

## Folder layout

```
Nemotron 3 Super/
├── nemotron_3_super.pdf      # the paper
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── ssm.py                # Mamba-2-style selective SSM layer (linear time)
│   ├── attention.py          # causal multi-head attention (quadratic)
│   └── model.py              # HybridModel: interleaves SSM & attention layers
├── data/
│   ├── generate_data.py      # long-range recall task generator
│   ├── recall_examples.json  # (generated) human-readable samples
│   └── hybrid_result.json    # (generated) accuracy + timing for the viz
├── demo/
│   └── run_demo.py           # trains the hybrid, times SSM vs attention
└── visualization/
    └── index.html            # hybrid stack + accuracy & linear-vs-quadratic charts
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

Trains the hybrid model and runs the timing sweep in ~38s on CPU, writing
`data/hybrid_result.json` for the visualization.

## Expected output

The demo asserts recall accuracy > 0.9 (it reaches 100%) and that attention's
time grows faster than the SSM's with length, then prints
`OK: hybrid Mamba-Attention learns long-range recall; SSM stays linear vs
attention's quadratic.`

## Explore the visualization

Open `visualization/index.html` in any browser (offline via `file://`): the
hybrid layer stack, the SSM recurrence, the measured recall-accuracy curve, the
measured SSM-vs-attention timing, and the asymptotic linear-vs-quadratic op
count. For live data, serve the folder:

```bash
python -m http.server 8000   # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper idea | Where | File |
|---|---|---|
| Mamba-2 selective SSM layer | linear recurrence | `src/ssm.py` → `SelectiveSSM` |
| Attention layer for global mixing | causal MHA | `src/attention.py` → `CausalAttention` |
| Hybrid Mamba-Attention layer pattern | interleaved stack | `src/model.py` → `HybridModel` |
| Long-context efficiency (linear vs quadratic) | timing sweep | `demo/run_demo.py` |
