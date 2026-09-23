# VibeVoice-ASR-Streaming — LLM-based streaming speaker-attributed ASR

A faithful, minimal, CPU-only reproduction of the core mechanism of
**VibeVoice-ASR-Streaming** (Tu et al., Microsoft Research, 2026). Everything
needed to read, run, and understand the idea lives in **this folder**: the paper
PDF, a from-scratch implementation, a runnable demo with synthesized audio, and
an interactive visualization.

- **Authors / org:** Yujie Tu, Zhiliang Peng, Jianwei Yu, Li Dong, Furu Wei, et al. — Microsoft Research (+ UCAS, SJTU)
- **Year:** 2026
- **Source:** arXiv:2609.02812 (`vibevoice-asr-streaming.pdf`, included here)

> **Honesty note.** The real system is a 1.5B/7B Qwen2.5-based Speech-LLM trained
> on thousands of hours of multi-speaker audio. We do **not** retrain that model.
> We reproduce, at small scale on CPU, the paper's central mechanism (§3.1):
> **interleaved speech-text streaming that emits speaker-attributed text
> incrementally, with a fixed lookahead and a chunk-end token, and no separate
> diarization stage.**

## What the demo shows

The paper's key claim is that a single autoregressive model, fed an interleaved
sequence `[X₁, Y₁, X₂, Y₂, …]` (speech chunk + lookahead, then speaker-attributed
text), can produce **“who said what” as speech arrives**. The demo reproduces
this end to end:

- **Two-speaker synthesized audio** — two distinct synthetic timbres (different
  pitch comb *and* spectral colour) alternate over time.
- A **tiny interleaved speech-text Transformer** trained (teacher forced) to emit,
  per chunk, speaker-labelled phone tokens ending in `<chunk_end>` — labelling
  speakers by **order of first appearance** and reusing labels via retained
  history (the diarization-free part).
- **Streaming decode**, chunk by chunk, printing the incremental transcript and
  reporting **transcription accuracy** and **speaker-attribution accuracy**.
- A **lookahead ablation**: each phone's identity lives in its *peak* frame, which
  falls in the lookahead, so removing the lookahead collapses boundary accuracy —
  mirroring Eq. 2 (L = 4 frames ≈ 0.5 s of future evidence).

```
VibeVoice-ASR-Streaming/
├── vibevoice-asr-streaming.pdf     # the paper
├── requirements.txt                # pinned CPU dependencies (torch, numpy)
├── src/
│   ├── audio.py                    # numpy 2-speaker synth + from-scratch STFT/log-mel
│   ├── data_gen.py                 # conversations + interleaved [speech, text] sequences
│   ├── model.py                    # tiny causal Transformer over mixed audio+text
│   └── streaming.py                # chunk-by-chunk decode + speaker-attributed metrics
├── demo/
│   └── run_demo.py                 # train + stream + lookahead ablation + write JSON
├── data/                           # generated JSON trace (for the visualization)
└── visualization/
    └── index.html                  # interleaved diagram + live "who said what" + accuracy
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

Trains and evaluates in about 25 s on CPU (single-threaded).

## 3. Expected output

- A training log (loss decreasing).
- A streaming **“who said what”** trace, one line per chunk, e.g.:

```
  t=[    0,  192] ms  chunk 0  ->  'Speaker0: aa ih \nSpeaker1: uw'
  t=[  192,  384] ms  chunk 1  ->  'aa \nSpeaker0: ao iy'
```

- Test-set metrics, roughly:

```
  with lookahead    : transcription  98.5%   speaker  99.3%
  without lookahead : transcription  43.9%   speaker  71.3%
  lookahead gain (transcription): +54.6 points
```

The script asserts streaming transcription and speaker attribution both exceed
90% with lookahead, and writes `data/streaming_run.json`.

## 4. Explore the visualization

Open `visualization/index.html` in any browser (works offline via `file://`,
with real demo data baked in). It shows the interleaved streaming sequence, a
play/step **“who said what”** timeline over the real log-mel spectrogram, and the
with/without-lookahead accuracy comparison. To load live data:

```bash
python -m http.server 8000   # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper section | Concept | File |
|---|---|---|
| §3.1, Eq. 1 | Interleaved sequence `[X₁, Y₁, X₂, Y₂, …]` | `src/data_gen.py`, `src/model.py` |
| §3.1, Eq. 2 | Fixed L-frame lookahead (future evidence) | `src/data_gen.py` (`chunk_audio_window`) |
| §3.1, Eq. 3 | `p(Yₖ | X<ₖ, X̃ₖ, Y<ₖ)` factorization | `src/model.py`, `src/streaming.py` |
| §3.1 | `<chunk_end>` token hands control to audio | `src/data_gen.py` (`CHUNK_END`) |
| §3.1 / App. A | Labels by order of first appearance, reused | `src/data_gen.py` (`gen_conversation`, `chunk_text`) |
| §3.1 | Retained history fixes speaker identity (no diarization) | causal context in `src/model.py` |
| §3 | Speech tokenizer / 16 kHz front-end | `src/audio.py` (from-scratch STFT + mel) |
