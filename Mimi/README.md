# Mimi — a streaming neural audio codec (from the Moshi tech report)

Mimi is the **streaming neural audio codec** introduced in *"Moshi: a
speech-text foundation model for real-time dialogue"* (Défossez et al., Kyutai,
2024; **arXiv:2410.00037**, included here as `mimi.pdf`). Mimi is the audio
**tokenizer** behind Moshi and behind **Hibiki**, the current state-of-the-art
for *simultaneous* speech-to-speech translation — which is why it is included in
this repo's [streaming audio encoder comparison](../docs/streaming-audio-encoders-for-speech-to-speech.md).

## What Mimi is (paper §3.3)

Mimi maps a 24 kHz waveform to **discrete tokens at 12.5 Hz** (~1.1 kbps) and
back, **fully streaming** (causal, 80 ms frame). Its defining ideas:

- A **SEANet-style causal convolutional** encoder/decoder — so it runs
  chunk-by-chunk in real time (§3.3.1).
- A **low frame rate (12.5 Hz)**, close to the ~3–4 Hz rate of text tokens, so a
  language model needs few autoregressive steps per second.
- **Residual vector quantization (RVQ)** — stacking codebooks gives a *scalable
  bitrate*: more codebooks reconstruct the signal more faithfully.
- A **split RVQ (§3.3.2):** codebook #1 is a **semantic** codebook trained by a
  cosine **distillation** loss toward a self-supervised **WavLM** representation,
  while a parallel **acoustic** RVQ carries fidelity/timbre. This fuses semantic
  and acoustic tokens in a single codec so downstream speech LMs get both
  intelligibility and voice.

## What this reproduction shows

`demo/run_demo.py` trains a *tiny* Mimi-style codec on synthetic "vowel" speech
(NumPy only) and verifies four properties, all on CPU in well under a minute:

| Property | What the demo shows | Result (reference run) |
|---|---|---|
| (a) Low frame rate | 2400 Hz → **12.5 Hz** tokens (×192), effective bitrate | 8 codebooks × 8 bits × 12.5 Hz = **0.80 kbps** |
| (b) Scalable bitrate | acoustic RVQ reconstructs the latent better with more codebooks | SNR **9.1 → 26.7 dB** (monotone) |
| (c) Streaming == offline | causal chunk-by-chunk == full pass | max diff **0.0e+00** |
| (d) Semantic distillation | codebook #1's cosine alignment to the semantic target | **−0.02 (untrained) → +0.57 (distilled)** |

### Honest scope

This is a small-scale CPU reproduction of Mimi's **mechanism**, not the trained
24 kHz codec. Two faithful simplifications (also noted in the code):

- **WavLM proxy.** There is no CPU-friendly WavLM, so codebook #1 is distilled
  toward a *transparent proxy* semantic target (a fixed embedding of the
  ground-truth vowel class). The distillation *loss* (cosine into codebook #1) is
  identical to the paper's; we report that the alignment rises far above an
  untrained baseline. Full phonetic separability of the semantic tokens needs the
  paper's WavLM teacher and model/data scale.
- **Scalable bitrate is measured on the RVQ latent reconstruction** (the
  quantizer's actual objective, guaranteed monotone), rather than full-band
  waveform fidelity, which a tiny CPU model cannot reach.

## Layout

```
Mimi/
├── mimi.pdf                     # the paper (Moshi tech report; introduces Mimi)
├── requirements.txt             # torch (CPU) + numpy
├── src/
│   ├── codec.py                 # §3.3.1 SEANet-style causal conv encoder/decoder (streaming)
│   ├── quantization.py          # §3.3.2 EMA VQ, ResidualVQ, split RVQ (semantic ∥ acoustic)
│   ├── model.py                 # §3.3   full codec + semantic distillation head
│   └── features.py              # hand-rolled STFT / multi-scale spectral loss (no torchaudio)
├── data/generate_audio.py       # synthetic "vowel" speech (numpy; no downloads)
├── demo/run_demo.py             # trains the tiny codec and checks (a)-(d)
└── visualization/index.html     # interactive, offline visualization (baked demo data)
```

## Setup & run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python demo/run_demo.py
# then open visualization/index.html in a browser
```

## Code ↔ paper map

| Paper (Moshi §3.3) | Code |
|---|---|
| §3.3.1 causal SEANet encoder/decoder, streaming | `src/codec.py` (`CausalConv1d`, `MimiEncoder`, `MimiDecoder`) |
| §3.3.1 12.5 Hz frame rate (striding 2·4·4·6·… ) | `MimiCodec(strides=…)`, `data/generate_audio.py` |
| §3.3.2 residual vector quantization (scalable bitrate) | `src/quantization.py` (`ResidualVQ`) |
| §3.3.2 split RVQ: semantic VQ ∥ acoustic RVQ | `src/quantization.py` (`SplitRVQ`) |
| §3.3.2 semantic distillation into codebook #1 | `MimiCodec.distillation_loss`, `demo/run_demo.py` |
| §3.3 multi-scale spectral (+ time) reconstruction | `src/features.py`, `demo/run_demo.py` |
