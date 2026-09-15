# FlashAttention

A faithful, minimal, self-contained reproduction of the core algorithm of Dao et
al., *"FlashAttention: Fast and Memory-Efficient Exact Attention with
IO-Awareness"* (2022, [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)):
compute **exact** attention block-by-block using an **online softmax**, so the
full `N × N` score matrix is never materialized.

## The core idea, in plain English

Standard attention computes the whole score matrix `S = QKᵀ` (one number per
query–key pair), softmaxes it, and multiplies by `V`. That matrix has `N²`
entries — for long sequences it dominates memory and the reads/writes to it
dominate runtime.

FlashAttention avoids ever building `S`. It streams over **blocks** of keys and
values and maintains, for each query row, two running numbers (paper §3.1,
Algorithm 1):

- `m` — the running maximum score seen so far, and
- `l` — the running sum of `exp(score − m)` (the softmax normalizer).

When a new block arrives, the partial output is **rescaled** by
`exp(m_old − m_new)` and the block's contribution is added. This "online softmax"
produces exactly the same result as computing the full softmax at once, but the
largest thing in memory is a single `block × block` tile.

## What the demo shows

`demo/run_demo.py` demonstrates two things:

1. **Exactness.** The tiled output matches naive full attention to
   floating-point precision — max absolute difference ~`3e-7` (and the same for
   the causal variant).

2. **Memory advantage.** Sweeping the sequence length `N`, it reports the peak
   score-matrix memory: naive grows as `O(N²)` while FlashAttention stays flat at
   one tile. At `N = 1024` the naive score matrix is `4 MB` vs. FlashAttention's
   `4 KB` — a **1024×** reduction.

Results are written to `data/flash_results.json` for the visualization.

> Note on wall-clock: on CPU with a plain Python tiling loop, FlashAttention is
> *not* faster than naive attention — its speed win comes from doing fewer reads
> and writes to slow GPU memory. This demo faithfully reproduces the **exactness**
> and the **memory** behavior, which are what make long-context attention
> feasible; it does not claim a CPU speedup.

## Folder layout

```
FlashAttention/
├── flashattention.pdf            # the paper
├── requirements.txt              # CPU PyTorch + numpy
├── src/
│   └── flash_attention.py        # naive attention + tiled online-softmax attention (Alg. 1)
├── data/
│   └── generate_data.py          # a tiny seeded (Q, K, V) sample
├── demo/
│   └── run_demo.py               # equivalence check + memory/time sweep
└── visualization/
    └── index.html                # tiling diagram + online-softmax + memory-vs-N chart
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
python data/generate_data.py     # writes a small sample Q,K,V to data/
python demo/run_demo.py          # equivalence + memory sweep; ~1s on CPU
```

## Expected output

```
max |naive - flash| = 2.98e-07  (floating-point noise)

       N |   naive mem |   flash mem |    ratio
      64 |     16.0 KB |      4.0 KB |       4x
     128 |     64.0 KB |      4.0 KB |      16x
     256 |    256.0 KB |      4.0 KB |      64x
     512 |      1.0 MB |      4.0 KB |     256x
    1024 |      4.0 MB |      4.0 KB |    1024x
```

## Explore the visualization

Open `visualization/index.html` in any browser (offline via `file://`):

- **Tiling diagram** — slide `N` and see the full `N×N` naive matrix next to the
  single block tile FlashAttention keeps in memory.
- **Online softmax** — the running-max / running-normalizer update, step by step.
- **Memory chart** — real peak-memory-vs-`N` on log-log axes: naive `O(N²)` vs.
  FlashAttention flat.

Serve over HTTP to load live JSON: `python -m http.server 8000`.

## Code ↔ paper map

| Paper section | Idea | File |
|---|---|---|
| §2 (background) | Naive attention materializes `N×N` | `src/flash_attention.py` (`naive_attention`) |
| §3.1, Algorithm 1 | Tiling + online softmax (running `m`, `l`) | `src/flash_attention.py` (`flash_attention`) |
| §3.1 | Exact same output as full attention | `demo/run_demo.py` (part a) |
| §1, §3 | IO-awareness → memory advantage | `demo/run_demo.py` (part b), `visualization/index.html` |
