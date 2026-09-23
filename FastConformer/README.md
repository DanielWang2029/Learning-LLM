# Fast Conformer

A faithful, minimal, CPU-only reproduction of the core idea in
**"Fast Conformer with Linearly Scalable Attention for Efficient Speech
Recognition"** — Dima Rekesh, Nithin Rao Koluguri, Samuel Kriman, Somshubra
Majumdar, Vahid Noroozi, He Huang, Oleksii Hrinchuk, Krishna Puvvada, Ankur
Kumar, Jagadeesh Balam, Boris Ginsburg (NVIDIA), 2023.
arXiv: [2305.05084](https://arxiv.org/abs/2305.05084).

## The idea in plain English

The Conformer encoder is the workhorse of modern speech recognition, but its
self-attention cost grows with the **square** of the sequence length. Conformer
starts with a convolutional *sub-sampling* front end that reduces the frame
rate by **4x** (from a 10 ms to a 40 ms hop). Fast Conformer's headline change
is to sub-sample by **8x** instead, using **depthwise-separable** convolutions:

- **8x downsampling** halves the number of tokens every attention/convolution
  block downstream must process. Because attention is O(T²), halving the tokens
  makes attention roughly **4x cheaper**.
- **Depthwise-separable convolutions** (a per-channel spatial filter plus a
  cheap 1×1 channel mixer) make the sub-sampling block *itself* much cheaper
  than the baseline's dense convolutions — the front end was ~20% of a large
  Conformer's compute.

The paper reports ~2.8x faster inference and ~2.9x fewer multiply-adds with no
accuracy loss. This repo demonstrates the same *mechanism* at tiny scale.

## What the demo shows

Two **identical** Conformer encoders are trained on a synthetic frame-labeling
task; the only difference is the sub-sampling front end (4x regular conv vs 8x
depthwise-separable conv). The demo prints, for each:

- the number of encoder **tokens** after sub-sampling,
- an analytic **multiply-add (MAC)** proxy for the sub-sampling block and for
  self-attention, and
- **frame accuracy** on held-out synthetic audio.

Expected: 8x produces ~half the tokens (→ ~4x cheaper attention) and a ~2x
cheaper sub-sampler, while keeping accuracy essentially equal to 4x.

## Folder layout

```
FastConformer/
├── fastconformer.pdf              # the paper
├── requirements.txt              # torch==2.8.0, numpy>=1.26 (CPU)
├── src/                          # from-scratch implementation
│   ├── subsampling.py            #   §2.1  4x regular & 8x depthwise-separable
│   └── conformer.py              #   minimal Conformer block + encoder
├── data/                         # numpy audio synthesizer + generated JSON
│   └── audio_synth.py            #   sine/chirp/formant/noise + manual log-mel
├── demo/
│   └── run_demo.py               # trains 4x vs 8x, reports tokens/MACs/accuracy
└── visualization/
    └── index.html                # interactive diagram + compute-vs-accuracy
```

## Setup

Reuse the shared CPU virtual environment (Python 3.10+):

```bash
source "/workspace/Attention Is All You Need/.venv/bin/activate"
```

Or create a fresh one from this folder:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## How to run

```bash
python demo/run_demo.py
```

Runs in a few seconds on one CPU thread. Flags such as `--epochs`,
`--num-blocks`, and `--num-segments` are available (`--help`).

## Expected output

```
metric                         4x baseline    8x FastConformer
----------------------------------------------------------------------
sub-sampling type             regular conv       depthwise-sep
encoder tokens                          20                  10
sub-sampling MACs                  545,280             254,040
attention MACs (proxy)             102,400              25,600
frame accuracy                       98.8%               99.1%

8x uses 2.00x fewer tokens than 4x -> 4.00x cheaper self-attention.
8x depthwise-separable sub-sampling is 2.15x cheaper than 4x regular-conv.
```

It also writes `data/demo_results.json`, which the visualization reads.

## Explore the visualization

Open `visualization/index.html` in any browser (works offline via `file://`)
for a color-coded encoder pipeline, the depthwise-separable math, and a live
compute-vs-accuracy comparison baked in from the demo. To load fresh results:

```bash
python -m http.server 8000    # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper section | Contribution | File |
|---|---|---|
| §2.1 (change #1) | 8x downsampling (3 strided layers) | `src/subsampling.py` → `DepthwiseSeparableSubsampling8x` |
| §2.1 (change #2) | Depthwise-separable convolutions | `src/subsampling.py` → `_DepthwiseSeparableConv` |
| §2.1 (change #4) | Reduced kernel size (9) | `src/subsampling.py` (`kernel_size=9`) |
| Baseline | 4x regular-conv sub-sampling | `src/subsampling.py` → `ConvSubsampling4x` |
| Conformer block | FFN / MHSA / Conv / FFN | `src/conformer.py` → `ConformerBlock` |
| §2 (efficiency) | Tokens & MAC comparison, 4x vs 8x | `demo/run_demo.py` |
