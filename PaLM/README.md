# PaLM

A faithful, minimal, and fully self-contained reproduction of the distinctive
architecture from **"PaLM: Scaling Language Modeling with Pathways"**
(Chowdhery et al., 2022, [arXiv:2204.02311](https://arxiv.org/abs/2204.02311)).
Everything needed to read, run, and understand the paper's *modeling* choices
lives in **this folder**: the paper PDF, a small from-scratch implementation, a
runnable CPU demo with data, and an interactive visualization.

## What PaLM is (plain English)

PaLM is a 540-billion-parameter decoder-only Transformer language model trained
on Google's Pathways system. Its headline result is scale, but the paper also
makes a handful of concrete, reusable **architecture changes** to the standard
Transformer decoder that have since become common in modern LLMs. This repo
reproduces exactly those changes in a tiny model you can train on a laptop:

- **SwiGLU activation** in the feed-forward network instead of ReLU.
- **Parallel layers**: attention and the MLP are computed from the *same*
  layer-normalized input and summed, rather than run one after the other.
- **Multi-Query Attention (MQA)**: all query heads share a single key/value
  head, which shrinks the decoding-time KV cache.
- **RoPE** rotary position embeddings instead of learned/absolute positions.
- **No biases** anywhere (dense layers and layer norms), for stability.

## What the demo shows

The demo builds a tiny PaLM, **prints confirmation that each of the five
features above is actually active** (with shapes and parameter counts), then
trains the model on a toy **copy task** — given `[BOS] c1..ck [SEP]`, produce
`c1..ck`. This requires moving information across positions (attention) and
knowing where each token is (RoPE), so success is real evidence the pieces work
together. It reaches **100% exact-copy accuracy in ~20 s on CPU** and writes a
loss curve for the visualization.

```
PaLM/
├── palm.pdf                  # the paper itself
├── requirements.txt          # pinned dependencies (CPU PyTorch)
├── src/                      # the paper's architecture, in code
│   ├── rope.py               #   RoPE rotary position embeddings
│   ├── normalization.py      #   LayerNorm without biases ("No Biases")
│   ├── feedforward.py        #   SwiGLU MLP
│   ├── attention.py          #   Multi-Query Attention (+ RoPE)
│   ├── block.py              #   PARALLEL attention + MLP block
│   └── model.py              #   full decoder-only PaLM language model
├── data/                     # sample dataset + generated demo artifacts
│   └── generate_demo_data.py #   writes a human-readable copy_dataset.json
├── demo/
│   └── run_demo.py           #   train + feature check + evidence + JSON export
└── visualization/
    └── index.html            # colorful, interactive PaLM block diagram
```

## 1. Set up the environment

Requires Python 3.10+. From inside this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

This installs the CPU build of PyTorch — no GPU needed. Everything runs on CPU.

## 2. (Optional) Generate a readable sample dataset

```bash
python data/generate_demo_data.py
```

Writes `data/copy_dataset.json`, a dozen concrete copy-task examples you can
inspect. The training demo generates batches on the fly, so this step is
optional.

## 3. Run the demo

```bash
python demo/run_demo.py
```

Expected output (abridged): a feature check where every PaLM feature reads
`[OK ]`, a loss curve dropping from ~2.8 toward ~0, exact-copy accuracy rising
to **100%**, and a worked example where the generated tokens match the content.
It also writes `data/palm_demo.json` for the visualization. All hyperparameters
are flags (`python demo/run_demo.py --help`).

## 4. Explore the visualization

Open `visualization/index.html` in any browser (works offline via `file://`)
for a color-coded, clickable PaLM block: click any component to see its role
and equations, compare the parallel vs. serial formulation, see how MQA shares
one K/V head, and view the **real loss curve** from your demo run baked into the
page. To load fresh data instead of the baked-in sample, serve the folder:

```bash
python -m http.server 8000    # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper (Section 2) | Component | File |
|---|---|---|
| "SwiGLU Activation" | SwiGLU feed-forward | `src/feedforward.py` |
| "Parallel Layers" | Parallel attention + MLP block | `src/block.py` |
| "Multi-Query Attention" | Shared single K/V head | `src/attention.py` |
| "RoPE Embeddings" | Rotary position embeddings | `src/rope.py` |
| "No Biases" | Bias-free LayerNorm & linears | `src/normalization.py`, `src/model.py` |
| Section 2 (model) | Full decoder-only LM | `src/model.py` |
