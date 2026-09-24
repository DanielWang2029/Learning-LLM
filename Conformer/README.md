# Conformer: Convolution-augmented Transformer for Speech Recognition

A faithful, minimal, self-contained reproduction of the **Conformer** encoder
block from Gulati et al., *"Conformer: Convolution-augmented Transformer for
Speech Recognition"* (2020), arXiv:2005.08100. Everything needed to read, run,
and understand the core idea lives in **this folder**: the paper PDF, a
from-scratch implementation, a runnable CPU demo with synthetic audio, and an
interactive visualization.

- **Authors:** Anmol Gulati, James Qin, Chung-Cheng Chiu, Niki Parmar, Yu Zhang, Jiahui Yu, Wei Han, Shibo Wang, Zhengdong Zhang, Yonghui Wu, Ruoming Pang (Google)
- **Year:** 2020 &nbsp; **arXiv:** [2005.08100](https://arxiv.org/abs/2005.08100)

## Plain-English summary

A Transformer's self-attention is great at capturing **global** context but is
weak at the **local**, fine-grained patterns that matter in speech. A CNN is the
opposite. The Conformer combines them: each block sandwiches a **self-attention**
module and a **convolution** module between two half-step **feed-forward**
modules (the "Macaron" structure), then applies a final LayerNorm. The
convolution module — pointwise conv → GLU → depthwise conv → BatchNorm → Swish →
pointwise conv — gives every channel a local temporal receptive field. This
combination set state-of-the-art word error rates on LibriSpeech.

## What the demo shows

We build a tiny Conformer encoder over **synthetic log-Mel "speech"** and train
it to classify, for every 10 ms frame, which "phoneme" (tone) is active. The
task is designed so that **both** modules are necessary:

- Each utterance starts with a **key** tone (a global fact stated once).
- The label of each later frame is `(local tone + key) mod K`.
- Recovering the label needs the **local** tone (convolution) **and** the
  **global** key from the start of the utterance (attention).

The demo trains the full model plus two ablations and prints the accuracy of
each. The full model reaches ~99% frame accuracy; removing either module drops
accuracy far toward chance — reproducing the paper's Section 3.4 finding that
convolution and attention are complementary.

## Folder layout

```
Conformer/
├── conformer.pdf              # the paper itself
├── requirements.txt           # pinned deps (CPU PyTorch + numpy)
├── src/                       # the paper, in code
│   ├── features.py            #   §3.1  hand-rolled log-Mel front end (STFT + Mel)
│   └── conformer.py           #   §2.2  FFN / MHSA / Conv modules, block, encoder
├── data/
│   └── generate_audio.py      #   numpy synthesizer for the tone task
├── demo/
│   └── run_demo.py            #   trains + ablates, writes data/demo_sample.json
└── visualization/
    └── index.html             #   interactive block diagram + real spectrogram
```

## Setup

Requires Python 3.10+. Reuse the shared virtual-env at the repo root, or create
one here:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## How to run

```bash
python demo/run_demo.py
```

Trains three tiny Conformer variants on CPU in well under a minute and prints the
ablation. It also writes `data/demo_sample.json` (a real spectrogram, per-frame
predictions, attention map, ablation numbers and accuracy curve) for the
visualization.

## Expected output

```
Conformer per-frame tone classification  (6 classes, chance 16.7%)
[full   attn+conv ]  frame-accuracy  99.x%
[ablate  attn-only ]  frame-accuracy  3x.x%   (no convolution module)
[ablate  conv-only ]  frame-accuracy  2x.x%   (no self-attention)
=> Convolution (local) and attention (global) are complementary (paper Section 3.4).
OK: Conformer learned the task and attention+convolution both help.
```

## Explore the visualization

Open `visualization/index.html` in any browser (works offline over `file://`)
for a clickable Conformer block, the convolution-module pipeline, the real
log-Mel spectrogram with ground-truth vs. predicted frames, the ablation bar
chart, the last-block attention map, and the training curve. To load fresh data
instead of the baked-in sample, serve the folder:

```bash
python -m http.server 8000   # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper section | Component | File |
|---|---|---|
| §3.1 | 80-ch filterbank front end (25 ms / 10 ms) | `src/features.py` |
| §2.2, Fig. 4 | Half-step feed-forward module (Swish, ×4) | `src/conformer.py` → `FeedForwardModule` |
| §2.2, Fig. 3 | Multi-head self-attention module | `src/conformer.py` → `MultiHeadSelfAttentionModule` |
| §2.2, Fig. 2 | Convolution module (PW → GLU → depthwise → BN → Swish → PW) | `src/conformer.py` → `ConvolutionModule` |
| §2.2, Fig. 1 | Conformer block (Macaron sandwich + LayerNorm) | `src/conformer.py` → `ConformerBlock` |
| §2.1 | Conformer encoder stack | `src/conformer.py` → `ConformerEncoder` |
| §3.4 | Convolution vs. attention ablation | `demo/run_demo.py` |

### Note on faithfulness

To stay minimal and CPU-friendly this reproduction uses standard (absolute)
self-attention rather than the relative positional attention of the paper, and
omits SpecAugment and the LSTM transducer decoder. The Conformer **block**
itself — the paper's contribution — is implemented exactly as described.
