# Cache-Aware Streaming Conformer

A faithful, minimal, CPU-only reproduction of the core idea in **"Stateful
Conformer with Cache-based Inference for Streaming Automatic Speech
Recognition"** — Vahid Noroozi, Somshubra Majumdar, Ankur Kumar, Jagadeesh
Balam, Boris Ginsburg (NVIDIA), 2023.
arXiv: [2312.17279](https://arxiv.org/abs/2312.17279).

## The idea in plain English

Streaming ASR must emit results as audio arrives, without reprocessing all past
audio for every new chunk. This paper makes a Conformer stream *exactly* by
carrying two caches across chunks:

- a **KV cache** — the self-attention **keys and values** of past frames, so a
  new chunk's queries attend over history without re-projecting it, and
- a **conv cache** — the last `K−1` inputs to each causal depthwise
  convolution, so the convolution spans the chunk boundary correctly.

Everything else (feed-forward, LayerNorm, 1×1 pointwise convs) is point-wise
and needs no cache. Convolutions are made **causal** and BatchNorm is replaced
by LayerNorm so nothing depends on future or global statistics. Attention uses
**chunk-aware look-ahead**, so a single model exposes a latency/accuracy
trade-off just by changing the chunk size at inference time.

The two headline guarantees:

1. **Streaming == offline.** Chunk-by-chunk cached inference produces the *same*
   output as one full-sequence forward pass.
2. **Constant per-chunk compute.** With a bounded left context, each streaming
   step attends over the same number of key frames — no past frame is
   recomputed.

## What the demo shows

One Conformer is trained with **dynamic chunk masking** (a random chunk size
per batch), then the demo:

1. **verifies** `max |full − streaming| < 1e-5` (exact caching),
2. **shows** per-chunk attended-key counts flatten to a constant once the KV
   cache is full (no recomputation),
3. **sweeps** the chunk size to trace a latency/accuracy trade-off from the one
   model — bigger chunk → more look-ahead → higher accuracy, higher latency.

To make look-ahead matter, the frame labels are *anticipatory* (each frame must
predict the class a few frames ahead), so future context genuinely helps.

## Folder layout

```
Cache-Aware Streaming Conformer/
├── cache-aware_streaming_conformer.pdf   # the paper
├── requirements.txt                     # torch==2.8.0, numpy>=1.26 (CPU)
├── src/
│   └── streaming_conformer.py           # KV cache, conv cache, chunk-aware attn
├── data/
│   └── audio_synth.py                   # numpy audio synth + manual log-mel
├── demo/
│   └── run_demo.py                      # equality + constant-compute + sweep
└── visualization/
    └── index.html                       # cache diagram + verification + sweep
```

## Setup

Reuse the shared CPU virtual environment (Python 3.10+):

```bash
source "/workspace/Attention Is All You Need/.venv/bin/activate"
```

Or create a fresh one:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## How to run

```bash
python demo/run_demo.py
```

Runs in ~15 s on one CPU thread. Flags include `--label-delay`,
`--left-context`, `--epochs` (`--help`).

## Expected output

```
[2/4] Verifying streaming (cached) == full-context forward ...
      chunk=8, max |full - streaming| = 2.7e-06  (< 1e-5 required)
[3/4] Per-chunk compute with bounded left context Lc=32 ...
      attended key frames per chunk: [8, 16, 24, 32, 32, 32, 32, 32, 32]
      -> constant at 32 once the cache is full (no recomputation of past frames).

mode               chunk     latency     frame acc
------------------------------------------------------------
chunk=2                2        20ms         75.5%
chunk=4                4        40ms         80.7%
chunk=8                8        80ms         84.8%
chunk=16              16       160ms         87.0%
full/offline          68       680ms         86.4%
```

It also writes `data/demo_results.json` for the visualization.

## Explore the visualization

Open `visualization/index.html` in any browser (offline via `file://`) for an
animated cache diagram (current chunk + KV/conv caches), the streaming==offline
verification, and the latency/accuracy sweep chart. For live data:

```bash
python -m http.server 8000    # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper section | Contribution | File / symbol |
|---|---|---|
| §3.3 | KV cache for self-attention | `src/streaming_conformer.py` → `StreamingAttention.forward_chunk` |
| §3.3 | Conv-state cache | `src/streaming_conformer.py` → `CausalConvModule.forward_chunk` |
| §3.1 | Causal conv + LayerNorm (streaming-safe) | `src/streaming_conformer.py` → `CausalConvModule` |
| §3.1 | Chunk-aware look-ahead mask | `src/streaming_conformer.py` → `build_chunk_mask` |
| §3.1 | Streaming == full-context inference | `demo/run_demo.py` (step 2) |
| §3.3 | Constant per-chunk compute | `demo/run_demo.py` (step 3) |
| §3.1 | One model, multiple latencies | `demo/run_demo.py` (step 4), dynamic chunk training |
