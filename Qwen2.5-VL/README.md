# Qwen2.5-VL

**Qwen Team / Alibaba, 2025 — [arXiv:2502.13923](https://arxiv.org/abs/2502.13923)**

Qwen2.5-VL makes high-resolution visual processing more efficient by using
window attention in most Vision Transformer layers and reserving full attention
for selected layers. It also targets spatial grounding, connecting language
descriptions to object locations.

This is a tiny CPU reproduction of the documented mechanisms, **not** the
released model. It learns colored 2×2 objects on a synthetic 8×8 native image
grid, predicting both object color and its normalized bounding box.

## What the demo shows

- Two 2×2-window attention blocks followed by one full-attention block.
- Exact attention-score-pair accounting versus three all-full layers.
- Object-color classification and a visual-grounding mean IoU on held-out
  synthetic images.

## Layout

```text
Qwen2.5-VL/
├── src/model.py              # window/full attention ViT and cost accounting
├── data/synthesizer.py       # NumPy object grids and bounding boxes
├── demo/run_demo.py          # training, evaluation, JSON trace
└── visualization/index.html  # offline architecture and results view
```

## Setup and run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python demo/run_demo.py
```

The expected output reports high color accuracy, a grounding IoU, and fewer
attention score pairs than all-full attention. It writes
`data/demo_results.json`. Open `visualization/index.html` via `file://`; it
contains baked results and can refresh them if served with `python -m http.server`.

## Code ↔ paper map

| Paper mechanism | Tiny reproduction | File |
|---|---|---|
| Window attention | Independent 2×2 attention windows | `src/model.py` |
| Sparse full attention | Final global attention layer connects all tokens | `src/model.py` |
| Efficiency | Explicit score-pair count (hybrid vs full) | `src/model.py` |
| Grounding | Normalized `(x0,y0,x1,y1)` regression head / IoU | `src/model.py`, `data/synthesizer.py` |
