# Emformer: Efficient Memory Transformer for Low-Latency Streaming ASR

A faithful, minimal, self-contained reproduction of the **Emformer** streaming
block from Shi et al., *"Emformer: Efficient Memory Transformer Based Acoustic
Model for Low Latency Streaming Speech Recognition"* (2021), arXiv:2010.10759.
Everything needed to read, run, and understand the core idea lives in **this
folder**: the paper PDF, a from-scratch implementation, a fast CPU demo on
synthetic audio features, and an interactive visualization.

- **Authors:** Yangyang Shi, Yongqiang Wang, Chunyang Wu, Ching-Feng Yeh, Julian Chan, Frank Zhang, Duc Le, Mike Seltzer (Facebook AI)
- **Year:** 2021 &nbsp; **arXiv:** [2010.10759](https://arxiv.org/abs/2010.10759)

## Plain-English summary

Full self-attention needs the whole utterance, which is impossible for
low-latency streaming. Emformer splits the utterance into **center blocks** and,
for each block, restricts attention to four bounded sources:

- an **augmented memory bank** — one summary vector per past block, distilling the
  entire history into a bounded set of vectors,
- the **left context** — the previous `L` frames (cached key/values, never
  recomputed),
- the **center** block — the `C` frames being emitted, and
- the **right context** — the next `R` frames (the lookahead that sets latency).

Because memory and left context are bounded, per-block cost is **independent of
utterance length**. A key trick (Section 2.2.2) is that the whole thing can be
computed in parallel at training time with a single masked attention, yet run
block-by-block at inference — and the two are identical.

## What the demo shows

Running a single Emformer layer over a synthetic feature sequence, the demo
verifies both claims:

1. **Streaming == parallel.** The block-by-block streaming pass (KV cache +
   memory bank) matches the parallel training-time pass to `~1e-7`. Each frame's
   key/value is computed **exactly once** (no recomputation) and the held state is
   **bounded** (`L + C + R + M` tokens), independent of the 118-frame length.
2. **Close to full context.** Emformer's bounded attention closely approximates a
   full bidirectional attention baseline, and the gap **shrinks as the left
   context grows** — the paper's accuracy-vs-latency trade-off.

## Folder layout

```
Emformer/
├── emformer.pdf               # the paper itself
├── requirements.txt           # pinned deps (CPU PyTorch + numpy)
├── src/
│   └── emformer.py            #   §2  EmformerBlock (parallel + streaming) + baseline
├── data/
│   └── generate_audio.py      #   numpy audio synth + from-scratch log-Mel features
├── demo/
│   └── run_demo.py            #   verifies equivalence + context sweep, writes JSON
└── visualization/
    └── index.html             #   interactive streaming/memory diagram + results
```

## Setup

Requires Python 3.10+. Reuse the shared virtual-env at the repo root, or:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## How to run

```bash
python demo/run_demo.py
```

Runs in a couple of seconds on CPU and writes `data/demo_sample.json` (the
equivalence numbers, the left-context sweep, and one block's attention map) for
the visualization.

## Expected output

```
[1] streaming vs. parallel (must match):
    max|Δ| = 1e-07   mean|Δ| = 1e-08   -> identical
    key/value computations = 118 for 118 frames (each frame once, no recomputation)
    peak cached state = 32 tokens (bounded: L + C + R + M, independent of length)
[2] Emformer (streaming) vs. full bidirectional attention:
    left-context sweep:  L=0 → 41%  ...  L=64 → 22%  of output scale (gap shrinks)
OK: streaming exactly reproduces full-context parallel output with bounded, reused state.
```

## Explore the visualization

Open `visualization/index.html` in any browser (offline `file://` works) for a
clickable diagram of block processing and the augmented memory bank, the
streaming-vs-parallel match, the bounded-state/no-recompute stats, the
left-context sweep chart, and a real attention map over
`[memory | left | center | right]`. To load fresh data:

```bash
python -m http.server 8000   # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper section | Component | File |
|---|---|---|
| §2.1 | Augmented memory bank (summary per block) | `src/emformer.py` → `EmformerBlock._summaries` |
| §2.1 | Left / center / right context attention + KV cache | `src/emformer.py` → `EmformerBlock.forward_stream` |
| §2.2.2 | Parallelizable block-processing mask | `src/emformer.py` → `EmformerBlock.forward_parallel` |
| §2.1 | Full-context baseline for comparison | `src/emformer.py` → `full_attention` |

### Note on faithfulness

To stay minimal and deterministic, memory summaries are the projected mean of a
block's center frames (rather than the paper's learned summary-query attention,
which also motivates the "disallow summary↔memory attention" rule of §2.2.3), and
the demo uses a single untrained layer to isolate the *attention mechanism*. The
streaming/parallel equivalence and bounded, reused state — the paper's core
contribution — are exact.
