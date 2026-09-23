# Zipformer

A faithful, minimal, CPU-only reproduction of the core ideas in
**"Zipformer: A faster and better encoder for automatic speech recognition"** —
Zengwei Yao, Liyong Guo, Xiaoyu Yang, Wei Kang, Fangjun Kuang, Yifan Yang,
Zengrui Jin, Long Lin, Daniel Povey, 2023 (ICLR 2024).
arXiv: [2310.11230](https://arxiv.org/abs/2310.11230).

## The idea in plain English

Zipformer redesigns the Conformer encoder to be both faster and more accurate.
The three ideas reproduced here:

1. **U-Net-like multi-rate structure (§3.1).** Instead of processing every
   layer at a fixed frame rate, Zipformer *downsamples* the sequence in the
   middle of the network, runs several blocks at a **lower frame rate**, then
   *upsamples* back. Since self-attention is O(T²), halving the frame rate in
   the middle makes those blocks ~4x cheaper — where most of the depth (and
   compute) lives.
2. **BiasNorm (§3.3).** A simpler replacement for LayerNorm:
   `BiasNorm(x) = x / RMS[x − b] · exp(γ)`. It drops the mean subtraction, uses
   a learnable bias `b` to *retain length information* (which LayerNorm removes),
   and a strictly-positive scale `exp(γ)` that avoids gradient-sign oscillation.
3. **Bypass (§3.2).** A learned channel-wise blend of a module's input and
   output, `(1 − c) ⊙ x + c ⊙ y`, initialized near "straight-through".

## What the demo shows

Two encoders with the **same number of blocks** are trained on a synthetic
frame-classification task:

- a **constant-rate** baseline (every block at the full frame rate), and
- a **Zipformer** whose middle blocks run at **half** the frame rate
  (downsample → low-rate blocks → upsample), with BiasNorm and Bypass.

The demo reports an analytic **self-attention FLOP proxy** and **frame
accuracy** for each. Expected: Zipformer cuts attention FLOPs by ~40% while
matching the baseline's accuracy.

## Folder layout

```
Zipformer/
├── zipformer.pdf                 # the paper
├── requirements.txt             # torch==2.8.0, numpy>=1.26 (CPU)
├── src/
│   └── zipformer.py             # BiasNorm, Bypass, down/upsample, U-Net encoder
├── data/
│   └── audio_synth.py           # numpy audio synth + manual log-mel
├── demo/
│   └── run_demo.py              # constant-rate vs Zipformer; FLOPs vs accuracy
└── visualization/
    └── index.html               # interactive U-Net + BiasNorm + FLOPs chart
```

## Setup

Reuse the shared CPU virtual environment (Python 3.10+):

```bash
source "/workspace/Attention Is All You Need/.venv/bin/activate"
```

Or create a fresh one:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## How to run

```bash
python demo/run_demo.py
```

Runs in ~15 s on one CPU thread. Flags include `--num-middle`, `--epochs`,
`--d-model` (`--help`).

## Expected output

```
metric                           constant-rate         Zipformer
----------------------------------------------------------------------
blocks                                       4                 4
middle frame rate                     full (T)        half (T/2)
attention FLOPs (proxy)              3,115,008         1,946,880
frame accuracy                          100.0%            100.0%

Zipformer cuts encoder attention FLOPs by 37.5% (1.60x fewer) while matching accuracy.
```

It also writes `data/demo_results.json` for the visualization.

## Explore the visualization

Open `visualization/index.html` in any browser (offline via `file://`) for an
interactive U-Net diagram (click any stage), the LayerNorm-vs-BiasNorm math,
and a FLOPs-vs-accuracy comparison baked in from the demo. For live data:

```bash
python -m http.server 8000    # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper section | Contribution | File / symbol |
|---|---|---|
| §3.1 | U-Net encoder, low-rate middle | `src/zipformer.py` → `ZipformerEncoder`, `downsample`, `upsample` |
| §3.2 | Bypass module | `src/zipformer.py` → `Bypass` |
| §3.3 | BiasNorm | `src/zipformer.py` → `BiasNorm` |
| §3.2 | Zipformer block | `src/zipformer.py` → `ZipformerBlock` |
| baseline | Constant-rate encoder | `src/zipformer.py` → `ConstantRateEncoder` |
| efficiency | FLOPs vs accuracy comparison | `demo/run_demo.py` |
