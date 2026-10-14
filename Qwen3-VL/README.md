# Qwen3-VL

**Qwen Team / Alibaba, 2025 — [arXiv:2511.21631](https://arxiv.org/abs/2511.21631)**

Qwen3-VL's DeepStack connects multi-level Vision Transformer features into
multiple language-model layers instead of giving the language model only one
final visual representation. The paper also describes interleaved M-RoPE.

This self-contained CPU companion focuses on DeepStack, **not** the released
Qwen3-VL model. It classifies the direction of a tiny two-pixel diagonal:
classes have identical global color averages, so fine local visual features
matter.

## What the demo shows

- A tiny ViT exposes low and mid patch-token features plus a pooled high level.
- A three-layer language-side stack receives low/mid/high visual features at
  different depths (DeepStack).
- The same stack using only the pooled final visual feature stays at chance on
  the fine-detail task, while multi-level injection learns it.

## Layout

```text
Qwen3-VL/
├── src/model.py              # tiny ViT, cross-attention language stack, DeepStack
├── data/synthesizer.py       # NumPy same-statistics diagonal images
├── demo/run_demo.py          # controlled baseline comparison and JSON trace
└── visualization/index.html  # offline interactive architecture/result view
```

## Setup and run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python demo/run_demo.py
```

Expected output shows multi-level accuracy above the 50% final-feature-only
baseline and writes `data/demo_results.json`. The standalone
`visualization/index.html` has baked data for `file://` and refreshes from that
JSON if served over HTTP.

## Code ↔ paper map

| Paper mechanism | Tiny reproduction | File |
|---|---|---|
| DeepStack | Low, mid, high ViT features injected at LM layers 1–3 | `src/model.py` |
| Multi-level vision features | ViT exposes patch-token and final pooled levels | `src/model.py` |
| Language-side fusion | Cross-attention from a decoder query into visual memory | `src/model.py` |
| Fine visual evidence | Two-pixel-diagonal synthetic controlled task | `data/synthesizer.py` |
