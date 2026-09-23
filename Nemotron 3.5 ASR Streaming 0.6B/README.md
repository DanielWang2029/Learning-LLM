# Nemotron 3.5 ASR Streaming 0.6B — cache-aware FastConformer-RNNT

A faithful, minimal, CPU-only reproduction of the core mechanisms of **Nemotron 3.5
ASR Streaming** (NVIDIA, 2026). Everything needed to read, run, and understand the
ideas lives in **this folder**: the model card, a from-scratch implementation, a
runnable demo with synthesized audio, and an interactive visualization.

- **Author / org:** NVIDIA
- **Year:** 2026 (Hugging Face release 2026-06-04)
- **Source:** official **model card** — there is no formal paper for this release.
  The card is included here as
  [`nemotron_35_asr_streaming_06b_model_card.md`](./nemotron_35_asr_streaming_06b_model_card.md)
  (model `nvidia/nemotron-3.5-asr-streaming-0.6b`). The underlying architecture
  builds on two papers cited by the card: *Stateful Conformer with Cache-based
  Inference* (arXiv:2312.17279) and *Fast Conformer* (arXiv:2305.05084).

> **Honesty note.** The real system is a 600M-parameter Cache-Aware FastConformer
> with 24 encoder layers and an RNN-T decoder, trained on 10k–1M hours of speech
> across 40 language-locales. We do **not** retrain that model. We reproduce, at
> tiny scale on CPU, the three headline mechanisms the model card documents:
> **cache-aware streaming, a configurable chunk-size latency/accuracy Pareto from a
> single model, and language-ID prompt conditioning.** Since there is no formal
> paper, the map below points to the model card rather than to equations.

## Plain-English summary

Streaming ASR must emit text before an utterance is finished. Naive "buffered"
streaming re-runs the encoder on overlapping windows, wasting computation.
Nemotron 3.5 is **cache-aware**: it processes each audio chunk exactly once and
reuses cached encoder state, so streaming output is identical to a full pass with
no redundant work. Crucially, the same trained model exposes a **chunk-size knob**
(80 / 160 / 320 / 560 / 1120 ms) chosen at inference — you pick your point on the
latency/accuracy curve without retraining. Finally, a **language-ID prompt** is
fused into the encoder so one multilingual model can be steered per-utterance.

## What the demo shows

A single tiny FastConformer-RNNT is trained (with a random right-context per step,
so one model serves every chunk size) on synthetic audio, then it demonstrates all
three properties:

- **(a) Cache-aware streaming == full context.** The streaming-assembled encoder
  output matches a single full-context pass to floating-point precision, because a
  frame's receptive field is bounded by `N·R` encoder frames (N layers × right
  context R) and the conv path is causal.
- **(b) Chunk-size Pareto from one model, no retraining.** Under a streaming
  emission-latency budget, larger chunks let the decoder integrate more of each
  (noisy) phone before committing → higher accuracy at higher latency.
- **(c) Language-ID prompt.** Two synthetic "languages" share acoustics but use
  different token sets; the prompt alone selects the correct transcription of
  *identical* audio.

```
Nemotron 3.5 ASR Streaming 0.6B/
├── nemotron_35_asr_streaming_06b_model_card.md   # the model card
├── requirements.txt                # pinned CPU dependencies (torch, numpy)
├── src/
│   ├── audio.py                    # numpy synth + from-scratch STFT / log-mel
│   ├── data_gen.py                 # two synthetic "languages" (prompt matters)
│   ├── model.py                    # FastConformer + lang fusion + RNN-T + loss
│   └── streaming.py                # cache-aware streaming, latency decode, WER
├── demo/
│   └── run_demo.py                 # train + (a) + (b) + (c) + write JSON
├── data/                           # generated JSON trace (for the visualization)
└── visualization/
    └── index.html                  # architecture + Pareto + language-ID prompt
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

Trains and evaluates in about 40 s on CPU (single-threaded).

## 3. Expected output

```
(a) cache-aware streaming vs full-context encoder output
  R= 0  max |streaming - full| = 7.15e-07   (identical up to fp precision)
  R= 3  max |streaming - full| = 7.15e-07   (identical up to fp precision)
  R=13  max |streaming - full| = 0.00e+00   (identical up to fp precision)

(b) chunk-size sweep: latency vs accuracy (one model, no retraining)
  offline full-context accuracy (unbounded latency): 94.6%
  R (80ms frames) | chunk (ms) | latency (ms) |     WER |     acc
                0 |         80 |           80 |   67.9% |   32.1%
                1 |        160 |          160 |   27.1% |   72.9%
                3 |        320 |          320 |    3.3% |   96.7%
                6 |        560 |          560 |    3.3% |   96.7%
               13 |       1120 |         1120 |    3.3% |   96.7%

(c) language-ID prompt on IDENTICAL audio
  prompt =  lang-A  ->  A3 A3 A2    (acc 100%)
  prompt =  lang-B  ->  B3 B3 B2    (acc 100%)
```

The script asserts streaming matches full context (< 1e-4), that the largest chunk
is at least 30 points more accurate than the smallest and exceeds 85%, and that
both language prompts exceed 80% accuracy; then it writes `data/streaming_run.json`.
(Exact numbers vary slightly with BLAS/thread nondeterminism.)

## 4. Explore the visualization

Open `visualization/index.html` in any browser (works offline via `file://`, with
real demo data baked in). It shows the FastConformer-RNNT architecture, the
streaming==full-context check, the interactive latency/accuracy Pareto (click a
chunk size), and the language-ID prompt switching the transcript on identical
audio. To load live data instead of the baked run:

```bash
python -m http.server 8000   # then open http://localhost:8000/visualization/
```

## Code ↔ model-card map

| Model-card concept | File |
|---|---|
| Cache-Aware FastConformer encoder, 8× subsampling | `src/model.py` (`SubsampleConv`, `FastConformerEncoder`) |
| Configurable right context R = chunk size, chosen at inference | `src/model.py` (`right_context_mask`), `demo/run_demo.py` (`R_SET`) |
| Cache-aware streaming == full context (no overlapping compute) | `src/streaming.py` (`streaming_encode`) |
| Language-ID prompt: broadcast one-hot → concat → project | `src/model.py` (`LangFusion`) |
| RNN-T decoder (prediction + joint) + transducer loss | `src/model.py` (`RNNT`, `transducer_loss`, `greedy_decode`) |
| Latency/accuracy Pareto via emission-latency-bounded decode | `src/streaming.py` (`latency_decode`), `demo/run_demo.py` |
| Language ID conditioning from a single model | `src/data_gen.py` (two token sets, shared acoustics) |
| 16 kHz front-end (from-scratch STFT + mel filterbank) | `src/audio.py` |
