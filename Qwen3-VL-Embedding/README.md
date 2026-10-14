# Qwen3-VL-Embedding and Qwen3-VL-Reranker

**Qwen Team / Alibaba, 2026 — [arXiv:2601.04720](https://arxiv.org/abs/2601.04720)**

Qwen3-VL-Embedding presents a unified multimodal embedding model: images and
text are encoded into one similarity space for retrieval. Qwen3-VL-Reranker
then jointly scores candidate image-text pairs to refine retrieval.

This is a tiny CPU reproduction of those documented mechanisms, **not** the
released model. It learns six synthetic color-and-shape image/caption concepts
with a symmetric InfoNCE dual encoder, then trains a small joint reranker.

## What the demo shows

- A shared-space image encoder and text encoder trained with symmetric InfoNCE.
- Image-to-caption zero-shot retrieval against all six captions.
- A joint pair scorer that reranks candidates after embedding similarity.
- A real 6×6 image-text similarity matrix written into the visualization JSON.

## Layout

```text
Qwen3-VL-Embedding/
├── src/model.py              # image/text dual encoders and joint reranker
├── data/synthesizer.py       # NumPy colored squares/diagonals and captions
├── demo/run_demo.py          # InfoNCE + reranker training and evaluation
└── visualization/index.html  # offline pipeline and similarity heatmap
```

## Setup and run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python demo/run_demo.py
```

The output reports retrieval and reranked top-1 accuracy (chance is 16.7%) and
writes `data/demo_results.json`. Open `visualization/index.html` directly; its
baked trace works with `file://` and it can load fresh JSON over a local HTTP
server.

## Code ↔ paper map

| Paper mechanism | Tiny reproduction | File |
|---|---|---|
| Unified multimodal embeddings | Normalized image and caption vectors | `src/model.py` |
| Contrastive training | Symmetric image→text / text→image InfoNCE | `demo/run_demo.py` |
| Retrieval | Dot-product similarity over all caption candidates | `demo/run_demo.py` |
| Reranker | Joint pixel + text-vector candidate-pair scorer | `src/model.py` |
