# Voxtral

A faithful, minimal, CPU-only reproduction of the core modeling idea in
**Voxtral** — Mistral AI's multimodal audio chat model (2025). Everything needed
to read, run, and understand the paper's audio pipeline lives in **this folder**:
the paper PDF, a from-scratch reference implementation, a runnable demo with
synthesized audio, and an interactive visualization.

- **Paper:** *Voxtral* (Mistral AI, 2025)
- **arXiv:** [2507.13264](https://arxiv.org/abs/2507.13264)

## What this reproduces

Voxtral is a Transformer with three parts (paper §2):

1. an **audio encoder** based on Whisper large-v3 (log-Mel → stride-2 conv stem →
   bidirectional Transformer, producing embeddings at a **50 Hz** frame rate),
2. an **audio-language adapter** that **concatenates 4 adjacent frames** and
   projects them with an MLP, downsampling **50 Hz → 12.5 Hz** (a 4× token
   reduction), and
3. a **language decoder** that autoregressively predicts text, conditioned on the
   audio embeddings as a soft-token prefix.

The reproduction implements all three from scratch and centers on the encoder +
adapter. The demo trains the pipeline on a synthetic **audio → token** task and
then **quantifies the 4× token reduction** and its effect on context length:
a 32k window holds ~11 minutes of audio at 50 Hz, but ~43 minutes after the
adapter — the mechanism that lets Voxtral handle 40-minute audio (§2.2).

## Folder layout

```
Voxtral/
├── voxtral.pdf                 # the paper
├── requirements.txt            # pinned CPU deps (torch + numpy only)
├── src/                        # reference implementation (the paper, in code)
│   ├── features.py             #   §2.1  from-scratch log-Mel front-end (numpy)
│   ├── encoder.py              #   §2.1  Whisper-style conv stem + bi-Transformer
│   ├── adapter.py              #   §2.2  4-frame-concat downsampling adapter
│   ├── decoder.py              #   §2    tiny causal language decoder
│   └── model.py                #   §2    full encoder→adapter→decoder pipeline
├── data/
│   └── synth_audio.py          # numpy audio synthesizer (formant stacks, no downloads)
├── demo/
│   └── run_demo.py             # end-to-end train + quantify 4× reduction (CPU, <60s)
└── visualization/
    └── index.html              # interactive pipeline + token-rate chart + spectrogram
```

## 1. Set up the environment

Requires Python 3.10+. From inside this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

This installs the CPU build of PyTorch, so everything runs on a laptop with no GPU.

## 2. Run the demo

```bash
python demo/run_demo.py
```

The script synthesizes audio (no downloads), computes log-Mel spectrograms,
trains the encoder + adapter + decoder to transcribe token sequences, and prints
the frame rate at each stage plus the token-reduction analysis. It writes
`data/demo_results.json` for the visualization.

### Expected output

Training reaches **100% token and sequence accuracy** in ~25–30 s on CPU, e.g.:

```
frames/clip: encoder(50Hz)=59  ->  adapter(12.5Hz)=14  (x4.21 fewer tokens ...)
...
Final token accuracy    : 100.0%
Final sequence accuracy : 100.0%

Example transcription
  target tokens : [3, 3, 9, 4, 11, 10]
  decoded tokens: [3, 3, 9, 4, 11, 10]

A 32k context holds ~10.7 min at 50 Hz, but ~42.7 min after the 4x adapter.
OK: encoder+adapter+decoder learned to transcribe, 4x token reduction confirmed.
```

(You may also run `python data/synth_audio.py` to write a small human-readable
`data/audio_preview.json` of token sequences and their durations.)

## 3. Explore the visualization

Open `visualization/index.html` in any browser (works offline via `file://`).
Click each stage of the pipeline to see its role and equations; read the real
token-rate reduction chart and context-length table; and inspect a real log-Mel
spectrogram and the training curve, all baked from the demo run.

For live (rather than baked-in) data, serve the folder so the page can fetch the
generated JSON:

```bash
python -m http.server 8000   # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper section | Component | File |
|---|---|---|
| §2.1 | log-Mel front-end (128 bins, hop 160) | `src/features.py` |
| §2.1 | Whisper encoder: conv stem (/2) + bi-Transformer, 50 Hz | `src/encoder.py` |
| §2.2 | Adapter: concat 4 frames + MLP, 50 → 12.5 Hz (4×) | `src/adapter.py` |
| §2   | Language decoder over audio-prefix + text | `src/decoder.py` |
| §2   | Full pipeline + greedy transcription | `src/model.py` |
| §2.1 | 30 s chunked processing (positions reset per chunk) | `src/encoder.py` (`sinusoidal_positions`) |
