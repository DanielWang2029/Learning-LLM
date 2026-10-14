# Qwen2-VL

**Qwen Team / Alibaba, 2024 — [arXiv:2409.12191](https://arxiv.org/abs/2409.12191)**

Qwen2-VL removes a fixed image-size assumption: native-size images become as
many visual tokens as their patch grid requires. Its Multimodal Rotary Position
Embedding (M-RoPE) splits rotary position signals into temporal, height, and
width components.

This is a tiny CPU reproduction of those documented mechanisms, **not** the
released Qwen2-VL model or its training data. It learns to locate one synthetic
colored square in one of four quadrants across several native resolutions.

## What the demo shows

1. 24×24, 32×32, and 40×40 images produce 36, 64, and 100 patch tokens without
   resizing.
2. A shared tiny vision transformer works across those variable sequence
   lengths.
3. M-RoPE's 2-D coordinates make the position-dependent task learnable; the
   otherwise identical position-free attention model remains near 25% chance.

## Layout

```text
Qwen2-VL/
├── src/model.py              # native patchification, 3-axis M-RoPE, attention
├── data/synthesizer.py       # NumPy colored-square generator
├── demo/run_demo.py          # end-to-end training and JSON trace
└── visualization/index.html  # offline interactive companion
```

## Setup and run

Requires Python 3.10+ and CPU PyTorch:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python demo/run_demo.py
```

The run writes `data/demo_results.json`. Expected evidence is high M-RoPE
quadrant accuracy and a near-chance no-position ablation. Open
`visualization/index.html` directly with `file://`; it includes baked demo data
and optionally refreshes it when served over HTTP.

## Code ↔ paper map

| Paper mechanism | Tiny reproduction | File |
|---|---|---|
| Naive Dynamic Resolution | Native 4×4 patch grids create variable-length sequences | `src/model.py` |
| M-RoPE | Separate rotary channel groups for time, height, width | `src/model.py` |
| Vision encoder | Two self-attention/MLP blocks | `src/model.py` |
| Visual input | Synthetic color-square images, no downloads | `data/synthesizer.py` |
