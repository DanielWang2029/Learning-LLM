# Whisper: Robust Speech Recognition via Large-Scale Weak Supervision

A faithful, minimal, self-contained reproduction of the **Whisper** encoder-decoder
from Radford et al., *"Robust Speech Recognition via Large-Scale Weak
Supervision"* (2022), arXiv:2212.04356. Everything needed to read, run, and
understand the model architecture lives in **this folder**: the paper PDF, a
from-scratch implementation, a fast CPU demo on synthetic audio, and an
interactive visualization.

- **Authors:** Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, Ilya Sutskever (OpenAI)
- **Year:** 2022 &nbsp; **arXiv:** [2212.04356](https://arxiv.org/abs/2212.04356)

## Plain-English summary

Whisper's headline result is *scale and robustness* — 680k hours of weakly
supervised audio — but its **architecture** is a clean, standard encoder-decoder
Transformer. Audio becomes an 80-channel log-Mel spectrogram; a small **conv
stem** (two layers, the second stride-2) halves the frame rate; a **Transformer
encoder** with sinusoidal positions processes the ~50 Hz features; and a
**Transformer decoder** autoregressively emits text tokens, attending to the
audio through cross-attention. Whisper reads a fixed **30-second** chunk
(non-streaming) and frames every task — transcription, translation, language id,
timestamps — as one sequence prediction problem steered by special **multitask
tokens**.

## What the demo shows

We build the full encoder-decoder and train it to **transcribe** synthetic
"speech": each utterance is a sequence of tone "words" from a 10-symbol
vocabulary, and the target transcript is the sequence of tone ids. The demo
prints token accuracy and exact-sequence accuracy rising to ~100%, then shows a
worked example where the decoded token sequence matches the audio.

## Folder layout

```
Whisper/
├── whisper.pdf                # the paper itself
├── requirements.txt           # pinned deps (CPU PyTorch + numpy)
├── src/
│   ├── features.py            #   §2.2  hand-rolled 80-bin log-Mel front end
│   └── whisper.py             #   §2.2  conv stem, AudioEncoder, TextDecoder, Whisper
├── data/
│   └── generate_audio.py      #   numpy synthesizer: tone "words" -> transcript
├── demo/
│   └── run_demo.py            #   trains seq2seq, writes data/demo_sample.json
└── visualization/
    └── index.html             #   interactive pipeline + real spectrogram + tokens
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

Trains on CPU in about half a minute and writes `data/demo_sample.json` (a real
spectrogram, the decoded vs. gold tokens, and the accuracy curve) for the
visualization.

## Expected output

```
Whisper-style speech recognition on synthetic audio (CPU)
vocab = 13 tokens (10 tone-words + SOT/EOT/PAD), model params = 746,496
step  200/320 | loss 0.036 | token-acc 100.0% | seq-acc 100.0%
Final token accuracy: ~100%   exact-sequence accuracy: ~100%
Example
  audio words (gold ids): [0, 7, 2, 5, 9, 4, 4, 4]
  decoded tokens        : [0, 7, 2, 5, 9, 4, 4, 4]
OK: Whisper-style encoder-decoder learned to transcribe the audio.
```

## Explore the visualization

Open `visualization/index.html` in any browser (offline `file://` works) for the
log-Mel → conv stem → encoder → decoder pipeline (clickable stages with
equations), the real 80-bin spectrogram, the decoded token strip vs. gold, and
the training accuracy curve. To load fresh data:

```bash
python -m http.server 8000   # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper section | Component | File |
|---|---|---|
| §2.2 | 80-bin log-Mel front end (25 ms / 10 ms) | `src/features.py` |
| §2.2 | Conv stem (2 layers, 2nd stride-2) | `src/whisper.py` → `AudioEncoder` |
| §2.2 | Sinusoidal positional encodings | `src/whisper.py` → `sinusoids` |
| §2.2 | Transformer audio encoder | `src/whisper.py` → `AudioEncoder` |
| §2.2 | Transformer text decoder (cross-attention) | `src/whisper.py` → `TextDecoder` |
| §2.3 | Multitask token prompt (SOT) | `demo/run_demo.py` |

### Note on faithfulness

This reproduction keeps Whisper's exact architecture (log-Mel, conv stem with
stride-2 downsample, sinusoidal positions, encoder-decoder Transformer with
cross-attention) but at tiny scale. It does not reproduce the 30 s padding, the
full multitask/multilingual token scheme, byte-pair tokenization, or the 680k-hour
training — the point here is the model, not the data-collection scale.
