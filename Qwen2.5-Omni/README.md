# Qwen2.5-Omni — Block-wise Streaming Audio Encoder

A faithful, minimal, CPU-only reproduction of the **streaming audio encoder** at
the heart of **Qwen2.5-Omni** (Alibaba, 2025). Everything needed to read, run,
and understand this part of the paper lives in **this folder**: the paper PDF, a
from-scratch reference implementation, a runnable demo with synthesized audio,
and an interactive visualization.

- **Paper:** *Qwen2.5-Omni Technical Report* (Alibaba, 2025)
- **arXiv:** [2503.20215](https://arxiv.org/abs/2503.20215)

## What this reproduces

Qwen2.5-Omni is an end-to-end multimodal model. To support **streaming**, it
"modified the audio and visual encoders to support block-wise attention along
the temporal dimension. Specifically, the audio encoder is changed from full
attention over the entire audio to performing attention in blocks of 2 seconds
each" (paper §2.4). This decouples **perception** (the encoder) from
**long-sequence modeling** (the LLM), enabling chunked prefill with bounded
latency.

This reproduction implements that block-wise audio encoder from scratch and
demonstrates its four defining properties on a synthetic per-frame recognition
task:

1. **Approximation** — with the same trained weights, 2-second block-wise
   attention gives *identical* downstream frame accuracy to full attention, and
   its embeddings converge to the full-attention ones as the block grows
   (cosine → 1).
2. **Streaming equivalence** — a block-diagonal masked pass is *mathematically
   identical* (max deviation ~1e-6) to encoding each 2 s block on its own, which
   is what makes streaming prefill possible.
3. **Cost** — attention drops from O(T²) to O(T·block); the demo shows up to 32×
   fewer attention pairs and a matching CPU speedup at long sequence lengths.
4. **Bounded latency** — per-block prefill time is constant regardless of the
   total audio length.

### High-level context (not implemented here)

- **TMRoPE** (Time-aligned Multimodal RoPE): a position encoding that gives audio
  one temporal id per **40 ms**, aligning audio, video and text on a shared time
  axis. The demo's encoder frame rate (40 ms/frame) matches this granularity.
- **Thinker–Talker**: the Thinker (an LLM) produces text/high-level
  representations; the Talker generates speech tokens from them. The block-wise
  encoder is the perception front-end feeding the Thinker.

## Folder layout

```
Qwen2.5-Omni/
├── qwen25-omni.pdf             # the paper
├── requirements.txt            # pinned CPU deps (torch + numpy only)
├── src/
│   ├── features.py             #   §2.1  from-scratch log-Mel front-end (numpy)
│   ├── encoder.py              #   §2.1/§2.4  block-wise attention audio encoder
│   └── model.py                #   §2.1  encoder + per-frame recognition head
├── data/
│   └── synth_audio.py          # numpy audio synthesizer (formant stacks, no downloads)
├── demo/
│   └── run_demo.py             # train + approximation/streaming/cost/latency (CPU, <60s)
└── visualization/
    └── index.html              # block-diagonal mask + approximation + cost + latency
```

## 1. Set up the environment

Requires Python 3.10+. From inside this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## 2. Run the demo

```bash
python demo/run_demo.py
```

It synthesizes audio (no downloads), trains the encoder with full attention, then
compares full vs block-wise attention and writes `data/demo_results.json` for the
visualization.

### Expected output

Training reaches 100% frame accuracy; block-wise (2 s) matches it, streaming is
equivalent, and cost scales linearly, e.g.:

```
  block=2.0s ( 50 fr) | frame-acc 100.0% | rel-L2 diff  15.4% | cosine 0.988
  full attention          | frame-acc 100.0% (reference)

Streaming equivalence: max|block-diagonal - per-block| = 9.54e-07 (IDENTICAL)

  T=1600 | attn-pairs full 2,560,000 vs block 80,000 (32.0x) | time 33.8ms vs 8.8ms

  per-block latency (ms): [...]  mean 0.44 ms, max 0.63 ms — bounded
OK: 2 s block-wise attention approximates full attention, is streaming-equivalent, and scales linearly.
```

## 3. Explore the visualization

Open `visualization/index.html` in any browser (offline via `file://`). It shows
the block-diagonal attention mask, the approximation-vs-block-size chart, the
O(T²)→O(T·block) cost comparison, and the bounded streaming-prefill latency — all
baked from the demo run. Serve the folder over http to load fresh data.

## Code ↔ paper map

| Paper section | Component | File |
|---|---|---|
| §2.1 | log-Mel front-end (128 bins, 25 ms/10 ms) | `src/features.py` |
| §2.1 | conv stem → 40 ms/frame encoder frames | `src/encoder.py` (`ConvStem`) |
| §2.4 | block-wise (block-diagonal) attention, 2 s blocks | `src/encoder.py` (`block_diagonal_mask`, `forward`) |
| §2.4 | streaming/chunked prefill (blocks on batch axis) | `src/encoder.py` (`streaming_encode`) |
| §2.1 | perception decoupled via per-frame head | `src/model.py` |
| §2.2 | TMRoPE (40 ms temporal id) — high level | this README + `FRAME_SECONDS` |
| §2.3 | Thinker–Talker — high level | this README |
