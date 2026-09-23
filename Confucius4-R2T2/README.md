# Confucius4-R2T2 — Longest Stable Prefix, append-only streaming ASR

A faithful, minimal, CPU-only reproduction of the **decoding paradigm** behind
**Confucius4-R2T2** ("R2T2 — Real Real-Time Transcription", NetEase Youdao,
2026). Everything needed to read, run, and understand the mechanism lives in
**this folder**: the model card, a from-scratch implementation, a runnable demo
with synthesized audio, and an interactive visualization.

- **Authors / org:** NetEase Youdao
- **Year:** 2026
- **Source:** *official model card — no formal paper yet* (a fine-tune of
  Qwen3-ASR; "tech report will be released soon"). This reproduction is grounded
  in the included `confucius4-r2t2_model_card.md`.

> **Honesty note.** The real model is a large multilingual Qwen3-ASR fine-tune
> trained with stable-prefix / forced-alignment data. We do **not** retrain that
> model. Instead we reproduce, at small scale and on CPU, the *documented
> mechanism* that makes it special: **true-streaming, append-only output via a
> Longest Stable Prefix (LSP)**. The underlying recognizer here is a transparent,
> training-free matched-filter classifier over synthetic phones, chosen so the
> LSP paradigm can be studied in isolation.

## What the demo shows

The model card's headline claims are that R2T2 (1) commits transcript text
**permanently without revising previous words** (no flicker), and (2) supports
**configurable chunks from 80 ms to 2 s** for different latency/accuracy
trade-offs. The demo reproduces both:

- A streaming recognizer whose **trailing edge is unstable** (a sound is
  ambiguous until it finishes) — exactly the situation LSP is designed for.
- **LSP** commits only the longest prefix on which consecutive hypotheses agree,
  minus a small `unfixed_token_num` rollback window → **append-only, 0 revisions**.
- A **naive** streaming decoder that re-displays the whole hypothesis → many
  **revisions** (flicker), despite identical final accuracy.
- A **chunk-size sweep** (80 ms → 2 s) from one recognizer with **no retraining**,
  showing the latency vs stability trade-off.

```
Confucius4-R2T2/
├── confucius4-r2t2_model_card.md   # the source model card
├── requirements.txt                # pinned CPU dependencies (torch, numpy)
├── src/                            # the mechanism, in code
│   ├── audio.py                    #   numpy synth + from-scratch STFT / log-mel
│   ├── recognizer.py               #   transparent matched-filter streaming ASR
│   └── lsp.py                      #   LSP append-only decoder + naive baseline
├── demo/
│   └── run_demo.py                 #   end-to-end streaming run + chunk sweep
├── data/                           # generated JSON trace (for the visualization)
└── visualization/
    └── index.html                  # interactive LSP mechanism + timeline + charts
```

## 1. Set up the environment

Requires Python 3.10+. From inside this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

This installs the CPU build of PyTorch (numpy does the signal processing; torch
is only used for the shared runtime contract with the other reproductions).

## 2. Run the demo

```bash
python demo/run_demo.py
```

Runs in well under a second on CPU. It synthesizes a ~2.4 s single-speaker
utterance of formant phones, streams it in 160 ms chunks through LSP and the
naive decoder, then sweeps chunk sizes from 80 ms to 2 s.

## 3. Expected output

- A per-chunk table where the **raw hypothesis** grows an unstable tail while the
  **committed prefix** only ever grows forward.
- `LSP revisions ... : 0` and `Naive decoder revisions (flicker) : 3`.
- Identical final transcripts (both 100% token accuracy) — the difference is
  stability, not accuracy.
- A sweep table, e.g. (representative):

```
chunk (ms) |  LSP acc | LSP rev | LSP lat (ms) | naive rev
        80 |   100.0% |       0 |        153.3 |         5
       160 |   100.0% |       0 |        260.0 |         3
       320 |   100.0% |       0 |        460.0 |         2
      2000 |   100.0% |       0 |       1200.0 |         0
```

The script asserts LSP made **0 revisions** and the naive decoder made **>0**,
and writes `data/streaming_run.json`.

## 4. Explore the visualization

Open `visualization/index.html` in any browser (works offline via `file://`,
with real demo data baked in). It shows the LSP commit mechanism, a play/step
**append-only vs revising** streaming timeline, and a latency/revision chart from
the chunk sweep. To load live data instead of the baked copy:

```bash
python -m http.server 8000   # then open http://localhost:8000/visualization/
```

## Code ↔ model-card map

| Model-card concept | Where it lives | File |
|---|---|---|
| "append-only output mode … without revising previous words" | `LSPDecoder` commits monotonically | `src/lsp.py` |
| Longest Stable Prefix (agreement across chunks) | `longest_common_prefix` of recent hypotheses | `src/lsp.py` |
| `unfixed_token_num` rollback window | `unfixed_token_num` (held-back trailing tokens) | `src/lsp.py` |
| Configurable 80 ms–2 s chunks | `CHUNK_SIZES_MS` sweep | `demo/run_demo.py` |
| Underlying streaming ASR (Qwen3-ASR) | transparent matched-filter stand-in | `src/recognizer.py` |
| 16 kHz audio front-end | from-scratch STFT + mel filterbank | `src/audio.py` |
| Pseudo-streaming baseline "may revise previously emitted text" | `NaiveDecoder` | `src/lsp.py` |
