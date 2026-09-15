# Scaling Embeddings Outperforms Scaling Experts in Language Models

A minimal, self-contained reproduction of the central claim of the Meituan
LongCat Team's *"Scaling Embeddings Outperforms Scaling Experts in Language
Models"* (2026), arXiv:**2601.21204**. Everything needed to read, run, and
understand the mechanism lives in this folder: the paper PDF, a from-scratch
PyTorch implementation, a runnable CPU demo, and an interactive visualization.

> **Honesty note.** This is a 2026 paper. The demo reproduces the paper's most
> characteristic *documented* mechanism — **N-gram Embedding (Over-Encoding)**
> as a way to scale capacity via the embedding, compared against scaling **MoE
> experts**, at a **fixed parameter budget** (paper §2–§3) — at tiny CPU scale
> on a synthetic task. It does **not** reproduce LongCat-Flash-Lite (68.5B), the
> 300B-token training, the per-layer embedding variants (§5), speculative
> decoding, or the reported benchmark numbers. It isolates and demonstrates the
> fixed-budget "embeddings vs experts" trade-off.

## What the paper is about

MoE is the standard way to grow model capacity cheaply, but it hits diminishing
returns and system bottlenecks. The paper argues the **embedding layer** is an
orthogonal, inherently sparse dimension for scaling: `O(1)` lookup, no routing
overhead. Concretely it uses **N-gram Embedding** (§2, Eq. 1–3), which augments
each token's vector with **hashed n-gram lookups** over the local context:

```
e_i = (1/((N-1)K+1)) · [ E0(t_i) + Σ_n Σ_k  W_{n,k} · E_{n,k}( H_{n,k}(t_{i-n+1..i}) ) ]
H_n(t_{i-n+1..i}) = ( Σ_{j=0}^{n-1} t_{i-j} · V0^j ) mod V_n          (rolling hash)
```

The key finding: at a **fixed parameter/compute budget**, allocating the extra
parameters to N-gram embeddings can achieve a better loss than allocating them
to more experts — especially in high-sparsity regimes.

## What the demo shows

`demo/run_demo.py` builds two tiny language models with the **same backbone**
(token embedding + one causal-attention layer + readout) and (within a fraction
of a percent) the **same total parameter count**, differing only in where the
extra ~98k parameters go:

- **scale-embeddings** — N-gram Embedding tables (here a collision-free trigram
  table, `hash_vocab = 16³`).
- **scale-experts** — a sparse MoE FFN with many top-k experts.

Both train on the same **trigram-grammar** task (each token's continuation
depends on the previous three tokens). The embedding-scaled model bakes the
predictive trigram straight into the input as a single lookup and converges to a
markedly lower loss / higher accuracy than the expert-scaled model, which must
compose a 3-way token interaction through attention + experts.

## Folder layout

```
Scaling Embeddings vs Experts/
├── scaling_embeddings_vs_experts.pdf   # the paper
├── requirements.txt                    # pinned deps (CPU PyTorch + numpy)
├── src/
│   ├── ngram_embedding.py              # §2  N-gram (Over-Encoding) embedding, Eq. 1-3
│   ├── moe.py                          # §3  sparse top-k MoE (the expert dimension)
│   ├── model.py                        #     1-layer causal LM with swappable capacity
│   └── task.py                         #     trigram-grammar dataset
├── data/
│   ├── generate_demo_data.py           # write a human-readable task sample
│   └── results.json                    # metrics + training curves (generated)
├── demo/
│   └── run_demo.py                     # fixed-budget embeddings-vs-experts comparison
└── visualization/
    └── index.html                      # interactive, offline: budget split + curves
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
python demo/run_demo.py                 # main demo (~19s on CPU)
python data/generate_demo_data.py       # optional: inspect the task
```

## Expected output

```
Fixed budget (total parameters, matched within a few %):
  scale-embeddings  104,817   (N-gram orders (3,), hash vocab 4096)
  scale-experts     105,248   (top-2/32 MoE experts)
  budget difference 0.4%

Final results on held-out sequences:
  allocation              params      loss    accuracy
  scale-embeddings       104,817    0.0032      100.0%
  scale-experts          105,248    0.1656       95.3%
```

Exact numbers vary slightly with the environment, but at a matched budget the
embedding-scaled model reliably reaches lower loss and higher accuracy than the
expert-scaled model. The demo exits non-zero if this does not hold.

## Explore the visualization

Open `visualization/index.html` in any browser (offline via `file://`). It shows
the fixed-budget split (shared backbone vs N-gram tables vs extra experts), the
N-gram Embedding equations, and the training curves for both allocations — all
from the real `data/results.json`. For live data, serve the folder:

```bash
python -m http.server 8000    # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper section | Concept | Where in code |
|---|---|---|
| §2, Eq. 1 | Augmented embedding e_i = base + Σ n-gram lookups | `NGramEmbedding.forward` |
| §2, Eq. 2 | Polynomial rolling hash H_n | `NGramEmbedding._hash` |
| §2, Eq. 3 | Over-Encoding normalization 1/((N-1)K+1) | `NGramEmbedding.scale` |
| §3 | Scaling experts (MoE) as the alternative dimension | `SparseMoE` in `src/moe.py` |
| §3 | Parameter-equivalent comparison (fixed budget) | `demo/run_demo.py` (matched totals) |
| §3.2.2 | Hash collisions vs vocabulary sizing | `hash_vocab = V0³` (collision-free trigram) |
