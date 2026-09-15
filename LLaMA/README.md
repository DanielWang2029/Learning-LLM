# LLaMA

A faithful, minimal, and fully self-contained reproduction of the architecture
from **"LLaMA: Open and Efficient Foundation Language Models"**
(Touvron et al., 2023, [arXiv:2302.13971](https://arxiv.org/abs/2302.13971)).
Everything needed to read, run, and understand the paper's *modeling* recipe
lives in **this folder**: the paper PDF, a small from-scratch implementation, a
runnable CPU demo with data, and an interactive visualization.

## What LLaMA is (plain English)

LLaMA is a family of decoder-only Transformer language models (7B–65B) trained
only on publicly available data. Its thesis: with enough good data and the
right recipe, comparatively *small* open models can match much larger ones —
LLaMA-13B outperforms GPT-3 (175B) on most benchmarks. Architecturally, LLaMA
is a standard decoder with three specific, now-canonical changes (paper
Section 2.2):

- **RMSNorm pre-normalization**: normalize the *input* of each sub-layer with
  RMSNorm (cheaper than LayerNorm, no mean-subtraction, no bias).
- **RoPE**: rotary position embeddings applied to queries/keys at every layer,
  instead of absolute position embeddings.
- **SwiGLU**: a gated feed-forward activation instead of ReLU.

This repo reproduces exactly this recipe in a tiny model you can train on CPU.

## What the demo shows

The demo trains the full LLaMA recipe on a toy **copy task** — given
`[BOS] c1..ck [SEP]`, produce `c1..ck` — reaching **100% exact-copy accuracy**.
It then runs a short **ablation study**: it retrains the same model with one
ingredient swapped at a time and reports the effect, proving each component is
genuinely wired in:

| Variant | Final copy accuracy |
|---|---|
| Full recipe (RMSNorm + RoPE + SwiGLU) | **100%** |
| RoPE off (no positions) | **~43%** — collapses |
| RMSNorm → LayerNorm | ~100% |
| SwiGLU → ReLU | ~100% |

Removing RoPE destroys the model's sense of order, so it can no longer copy —
concrete evidence that positions flow through the rotary path. (RMSNorm and
SwiGLU are quality/efficiency choices; both alternatives still train on this
easy task, which is exactly what you'd expect.)

```
LLaMA/
├── llama.pdf                 # the paper itself
├── requirements.txt          # pinned dependencies (CPU PyTorch)
├── src/                      # the paper's recipe, in code
│   ├── rope.py               #   RoPE rotary position embeddings
│   ├── normalization.py      #   RMSNorm
│   ├── feedforward.py        #   SwiGLU (+ ReLU baseline for ablation)
│   ├── attention.py          #   causal multi-head attention (+ RoPE)
│   ├── block.py              #   pre-norm block: x + Attn(norm x) + FFN(norm x)
│   └── model.py              #   full decoder-only LLaMA (with ablation switches)
├── data/                     # sample dataset + generated demo artifacts
│   └── generate_demo_data.py #   writes a human-readable copy_dataset.json
├── demo/
│   └── run_demo.py           #   train + ablation study + JSON export
└── visualization/
    └── index.html            # colorful, interactive LLaMA block diagram
```

## 1. Set up the environment

Requires Python 3.10+. From inside this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

CPU build of PyTorch — no GPU needed.

## 2. (Optional) Generate a readable sample dataset

```bash
python data/generate_demo_data.py
```

Writes `data/copy_dataset.json`. The training demo generates batches on the
fly, so this is optional.

## 3. Run the demo

```bash
python demo/run_demo.py
```

Expected output: the full recipe converging to **100%** exact-copy accuracy,
followed by the ablation table above. Total runtime is ~30 s on CPU. It writes
`data/llama_demo.json` for the visualization. All hyperparameters are flags
(`python demo/run_demo.py --help`).

## 4. Explore the visualization

Open `visualization/index.html` in any browser (offline via `file://`) for a
color-coded, clickable LLaMA block, the "small open model beats big model"
thesis, the real training-loss curve, and the ablation bar chart — all baked in
from your demo run. To load fresh data, serve the folder:

```bash
python -m http.server 8000    # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper (Section 2.2) | Component | File |
|---|---|---|
| "Pre-normalization [GPT3]" | RMSNorm on sub-layer inputs | `src/normalization.py`, `src/block.py` |
| "SwiGLU activation function [PaLM]" | SwiGLU feed-forward | `src/feedforward.py` |
| "Rotary Embeddings [RoPE]" | Rotary positions on Q/K | `src/rope.py`, `src/attention.py` |
| Section 2.2 (model) | Full decoder-only LM | `src/model.py` |
