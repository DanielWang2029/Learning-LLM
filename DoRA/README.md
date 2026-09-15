# DoRA — Weight-Decomposed Low-Rank Adaptation

A faithful, minimal, CPU-only reproduction of *DoRA* (NVIDIA, 2024 — arXiv
[2402.09353](https://arxiv.org/abs/2402.09353)). Everything needed to read, run,
and understand the method lives in this folder: the paper PDF, a from-scratch
implementation, a runnable demo with data, and an interactive visualization.

DoRA improves on LoRA by first **decomposing** each pretrained weight into a
per-output **magnitude** vector `m` and a **direction** matrix `V`, then adapting
them separately (paper Eq. 3):

```
W0 = m0 · V0 / ||V0||          (m0 = per-output norm of W0,  V0 = W0)
W  = m  · (V0 + ΔV) / ||V0 + ΔV||        with ΔV = (α/r) · B·A
```

The magnitude `m` is trained directly, while the **direction** receives a
low-rank LoRA update. Decoupling "how big" from "which way" gives DoRA learning
dynamics closer to full fine-tuning than LoRA — its central result.

```
DoRA/
├── dora.pdf                 # the paper
├── requirements.txt         # torch (CPU) + numpy
├── src/
│   ├── model.py             # a tiny MLP backbone to pretrain and adapt
│   ├── adapters.py          # §4  LoRALinear and DoRALinear + injection helpers
│   └── data.py              # shared-feature base / new toy tasks
├── data/
│   └── generate_data.py     # writes a few labeled samples of each task
├── demo/
│   └── run_demo.py          # pretrain -> full-FT vs LoRA vs DoRA, compare
└── visualization/
    └── index.html           # decomposition diagram, accuracy bars, curves
```

## 1. Set up the environment

Requires Python 3.10+. From inside this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## 2. Peek at the data (optional)

```bash
python data/generate_data.py
```

Writes `data/samples.json` with a few labeled examples from the base and new
tasks (both built on a shared feature extractor, with different heads).

## 3. Run the demo

```bash
python demo/run_demo.py
```

Runs end-to-end on CPU in under 10 seconds. It pretrains and freezes a base
model, then adapts it to a new task three ways — full fine-tuning (upper bound),
LoRA, and DoRA — at rank `r=1`, and writes `data/dora_results.json`.

### Expected output

```
full fine-tuning :  75,782 params ->  83.6%  (upper bound)
DoRA (r=1)       :   1,580 params ->  71.2%
LoRA (r=1)       :   1,062 params ->  57.7%
gap to full-FT   : DoRA +12.3  |  LoRA +25.8
DoRA beats LoRA by +13.5 points at ~matched budget
```

DoRA's only extra parameters over LoRA are the magnitude scalars (one per output
unit — here 518 total), yet it closes most of the gap to full fine-tuning while
LoRA lags well behind. (Numbers are deterministic given the fixed seeds.)

## 4. Explore the visualization

Open `visualization/index.html` in any browser (works offline via `file://`).
It shows the magnitude/direction decomposition diagram, the head-to-head accuracy
bars with parameter counts, the real accuracy-over-training curves for all three
methods, and a chart of how DoRA moved magnitude vs direction per output unit
(the extra degree of freedom LoRA lacks).

To load live data instead of the baked-in snapshot, serve the folder:

```bash
python -m http.server 8000    # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper idea (§4) | Component | File |
|---|---|---|
| Weight decomposition `W = m · V/‖V‖` | `DoRALinear.effective_weight` | `src/adapters.py` |
| Magnitude `m` trained directly | `DoRALinear.magnitude` (init = `‖W0‖`) | `src/adapters.py` |
| Low-rank update on the direction | `ΔV = (α/r)·B·A` | `src/adapters.py` |
| LoRA baseline `W = W0 + (α/r)·B·A` | `LoRALinear` | `src/adapters.py` |
| Matched-budget comparison | trainable-param accounting | `demo/run_demo.py` |
| DoRA tracks full-FT better than LoRA | held-out accuracy comparison | `demo/run_demo.py` |
