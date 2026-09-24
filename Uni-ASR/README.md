# Uni-ASR

A faithful, minimal, CPU-only reproduction of the core ideas in **Uni-ASR:
Unified LLM-Based Architecture for Non-Streaming and Streaming Automatic Speech
Recognition** (Alibaba, 2026). Everything needed to read, run, and understand the
paper's streaming mechanism lives in **this folder**: the paper PDF, a
from-scratch reference implementation, a runnable demo with synthesized audio,
and an interactive visualization.

- **Paper:** *Uni-ASR* (Alibaba, 2026)
- **arXiv:** [2603.11123](https://arxiv.org/abs/2603.11123)

## What this reproduces

Uni-ASR is a single model — **Conformer encoder + adapter + LLM decoder** (§2.1) —
jointly trained for both non-streaming and streaming ASR. Two ideas make that
work:

1. **Interleaved speech-text training with loss masks** (§2.2.2). Streaming is
   framed as multiple rounds of non-streaming decoding over an interleaved
   sequence `[a0, t0, a1, t1, ...]`, where each speech segment `a_j` is followed
   by its text token, and the loss is taken only on the text targets.
2. **Latest-token fallback decoding** (§2.3). Near a chunk boundary a token's
   acoustic evidence has not fully arrived, so decoding it is a guess. The last
   token of each chunk is emitted *provisionally* and **re-decoded when the next
   chunk arrives** — improving streaming accuracy at **no added latency**. A
   **context-aware (CS) training** masks that boundary token during training so
   the model learns to re-supply it.

This reproduction implements all three components from scratch (with a **causal**
Conformer so streaming has no future leakage) and demonstrates the payoff on a
synthetic audio → token task where each token's identity lives in its *nucleus*
(second half), creating genuine boundary ambiguity.

### The result

One model, three decoding modes on the same held-out audio:

| Mode | Token accuracy |
|---|---|
| Non-streaming (full audio) | **100%** |
| Streaming, naive chunked | 18–100% (worse as chunks misalign with tokens) |
| Streaming, **latest-token fallback** | **100%** at every chunk size |

The fallback cuts **mean streaming token error from ~38% to 0%** at equal
latency. Chunk sizes that are exact multiples of the 160 ms token length never
split a token, so naive already has no error there — a nice sanity check that the
errors are specifically boundary-induced.

## Folder layout

```
Uni-ASR/
├── uni-asr.pdf                 # the paper
├── requirements.txt            # pinned CPU deps (torch + numpy only)
├── src/
│   ├── features.py             #   from-scratch log-Mel front-end (numpy)
│   ├── conformer.py            #   §2.1  causal Conformer encoder
│   └── model.py                #   §2.1-2.3  adapter + LLM decoder, interleaving, fallback
├── data/
│   └── synth_audio.py          # numpy audio synthesizer (onset + identifying nucleus)
├── demo/
│   └── run_demo.py             # joint train + naive-vs-fallback streaming (CPU, <60s)
└── visualization/
    └── index.html              # interleaved sequence + error-by-chunk + worked example
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

It synthesizes audio (no downloads), jointly trains the one model for
non-streaming + streaming, then decodes each held-out clip three ways and writes
`data/demo_results.json` for the visualization.

### Expected output

```
Non-streaming accuracy (full audio): 100.0%

Streaming: naive vs latest-token fallback, by chunk size (equal latency)
     chunk |  naive acc | fallback acc | naive err | fallback err
     80ms |      18.2% |       100.0% |     81.8% |         0.0%
    120ms |      31.2% |       100.0% |     68.8% |         0.0%
    160ms |     100.0% |       100.0% |      0.0% |         0.0%
    ...
  mean streaming error — naive 37.9%  vs  fallback 0.0%
  fallback cuts streaming token error by 100% (relative)
OK: one model does non-streaming + streaming; fallback cuts streaming errors at equal latency.
```

## 3. Explore the visualization

Open `visualization/index.html` in any browser (offline via `file://`). It shows
the interleaved speech-text sequence, the streaming-error-by-chunk-size chart
(naive vs fallback), and a worked example where naive garbles boundary tokens and
fallback recovers the exact transcript. Serve the folder over http to load fresh
data.

## Code ↔ paper map

| Paper section | Component | File |
|---|---|---|
| §2.1 | log-Mel front-end | `src/features.py` |
| §2.1 | Conformer audio encoder (causal for streaming) | `src/conformer.py` |
| §2.1 | adapter (two linear + ReLU) | `src/model.py` (`Adapter`) |
| §2.1 | LLM decoder | `src/model.py` (`DecoderLM`) |
| §2.2.2 | interleaved speech-text training + loss masks | `src/model.py` (`training_batch`) |
| §2.2.2 | streaming = growing audio prefix / KV accumulation | `src/model.py` (`decode_stream`) |
| §2.3 | latest-token fallback re-decode | `src/model.py` (`decode_stream`, `fallback=True`) |
| §2.3 | context-aware (CS) boundary-token masking | `src/model.py` (`training_batch`, `context_aware`) |
