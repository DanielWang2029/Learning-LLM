# ZAYA1-8B — Compressed Convolutional Attention (CCA)

A minimal, self-contained reproduction of the core attention mechanism from the
Zyphra team's *"ZAYA1-8B Technical Report"* (2026), arXiv:**2605.05365**.
Everything needed to read, run, and understand the mechanism lives in this
folder: the paper PDF, a from-scratch PyTorch implementation, a runnable CPU
demo, and an interactive visualization.

> **Honesty note.** This is a 2026 paper. The demo reproduces the report's most
> characteristic *documented* architectural mechanism — **Compressed
> Convolutional Attention (CCA)**, in which a lightweight convolution compresses
> keys and values into a shorter latent sequence before attention (paper
> §II-A-1) — at tiny CPU scale on a synthetic task. It does **not** reproduce
> the 8B-parameter model, its MoE routing, the 12T-token pretraining, or any
> reported benchmark numbers. It isolates and demonstrates the CCA compression
> vs. full-attention quality/KV-cache trade-off.

## What the paper is about

ZAYA1-8B is an efficient dense/MoE language model whose headline attention block
is **Compressed Convolutional Attention**. Standard attention lets every query
attend over all `L` key/value positions, so both the attention matrix and the
decoding **KV-cache** grow with sequence length. CCA inserts a strided
(depthwise) convolution that both *mixes local context* and *downsamples* the
key/value sequence by a factor `r` before attention runs:

```
K, V   = projections of x                          # (B, L, d)
K_c, V_c = Conv1d_stride_r(K), Conv1d_stride_r(V)  # (B, L/r, d)   -- compress
attn   = softmax(Q · K_cᵀ / √d_head)               # (B, L, L/r)
out    = attn · V_c
```

Queries attend over only `L/r` compressed positions, so the attention matrix and
the KV-cache both shrink by `r`, while (the paper reports) quality is preserved
because each compressed token summarizes a whole window of the original
sequence.

## What the demo shows

`demo/run_demo.py` trains three tiny models on a content-based retrieval task
(each sequence has one *marked* token; the label is that token's payload id — the
canonical job of attention) and compares them on CPU in under a minute:

- **full attention** — the baseline (KV-cache = `L` positions).
- **CCA, compress = 4** — conv-downsample K/V by 4 (KV-cache = `L/4`).
- **CCA, compress = 8** — conv-downsample K/V by 8 (KV-cache = `L/8`).

It reports accuracy, KV-cache size (positions and scalars), attention-matrix
size, and how often CCA agrees with the full model, then verifies that CCA
matches full-attention quality while shrinking the KV-cache by the compression
factor.

## Folder layout

```
ZAYA1-8B/
├── zaya1-8b.pdf                # the paper
├── requirements.txt            # pinned deps (CPU PyTorch + numpy)
├── src/
│   ├── cca.py                  # §II-A-1  FullAttention, CompressedConvAttention, Classifier
│   └── task.py                 #          marked-token retrieval task
├── data/
│   ├── generate_demo_data.py   # write a human-readable task sample
│   └── results.json            # metrics + attention maps written by the demo (generated)
├── demo/
│   └── run_demo.py             # end-to-end CPU demo (full vs CCA-4 vs CCA-8)
└── visualization/
    └── index.html              # interactive, offline: CCA pipeline + savings
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
python demo/run_demo.py                 # main demo (~45s on CPU)
python data/generate_demo_data.py       # optional: inspect task sequences
```

## Expected output

```
ZAYA1-8B — Compressed Convolutional Attention (CCA)
Task: retrieve the marked token's payload. seq_len=48, vocab=16.

Results on held-out sequences:
  model       accuracy  KV positions   KV scalars  attn entries  agree w/ full
  full          100.0%            48         4608          2304          100.0%
  cca-4         100.0%            12         1152           576          100.0%
  cca-8          99.9%             6          576           288           99.9%

KV-cache reduction:  cca-4 = 4x,  cca-8 = 8x  smaller than full.

OK: CCA matches full-attention quality while shrinking the KV-cache 8x.
```

Exact numbers vary slightly with the environment, but CCA matches full
attention's accuracy (≈100%) while the KV-cache and attention matrix shrink by
the compression factor. The demo exits non-zero otherwise.

## Explore the visualization

Open `visualization/index.html` in any browser (offline via `file://`). It shows
the CCA pipeline diagram and equations, the accuracy vs. KV-cache trade-off
across full / CCA-4 / CCA-8, and side-by-side attention heatmaps (full attention
over `L` positions vs. CCA-8 over `L/8` compressed positions) — all from the real
`data/results.json`. For live data:

```bash
python -m http.server 8000    # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper section | Concept | Where in code |
|---|---|---|
| §II-A-1 | Compressed Convolutional Attention (compress K/V, then attend) | `CompressedConvAttention` in `src/cca.py` |
| §II-A-1 | Strided depthwise conv as the K/V downprojector | `conv_k`, `conv_v` in `CompressedConvAttention` |
| — | Standard attention baseline | `FullAttention` in `src/cca.py` |
| §II-A-1 | KV-cache / attention-matrix savings by factor `r` | `kv_positions`, `kv_cache_size` in `src/cca.py` |
| — | Retrieval task that requires attention | `MarkedRetrieval` in `src/task.py` |
| — | Quality-vs-compression comparison | `demo/run_demo.py` |
