# CLIP — Learning Transferable Visual Models From Natural Language Supervision

A faithful, minimal, fully self-contained reproduction of **CLIP** from Radford
et al., *"Learning Transferable Visual Models From Natural Language Supervision"*
(2021), arXiv:[2103.00020](https://arxiv.org/abs/2103.00020). Everything needed
to read, run, and understand the core idea lives in **this folder**: the paper
PDF, a from-scratch implementation, a runnable demo with synthetic data, and an
interactive visualization.

## The core idea, in plain English

CLIP learns from (image, caption) pairs instead of hand-labeled classes. It
trains **two encoders** — one for images, one for text — to place a matching
image and caption close together in a shared vector space, and mismatched pairs
far apart. The training signal is purely *contrastive*: within a batch of `N`
pairs, the `N` correct (image, text) matches lie on the diagonal of an `N × N`
similarity matrix, and a **symmetric cross-entropy (InfoNCE)** loss pushes every
image toward its own caption and every caption toward its own image.

Because the model understands *language*, not a fixed label set, it can classify
images **zero-shot**: to recognize a new image, embed candidate captions like
`"a photo of a red circle"` and pick the closest one — no fine-tuning, no labels.

## What the demo shows

We can't download web images, so we synthesize a tiny compositional analogue:
small RGB images of a colored geometric shape (4 colors × 3 shapes = **12
classes**) with captions built from the labels. The demo:

1. trains the dual encoder with the symmetric contrastive loss, then
2. classifies **held-out** images zero-shot by nearest caption, reaching
   **~99% accuracy vs. 8.3% chance**, and
3. prints an image↔text similarity matrix with a clean bright diagonal.

## Folder layout

```
CLIP/
├── clip.pdf                  # the paper itself
├── requirements.txt          # pinned CPU dependencies (torch + numpy)
├── src/                      # the paper, in code
│   ├── tokenizer.py          #   fixed-vocab word tokenizer for captions
│   ├── encoders.py           #   §2.4  image encoder (CNN) + text encoder
│   ├── model.py              #   §2    dual encoder, shared space, temperature
│   └── loss.py               #   §2.2  symmetric InfoNCE (Figure 3)
├── data/
│   └── shapes.py             #   numpy generator: colored-shape images + captions
├── demo/
│   └── run_demo.py           #   train → zero-shot eval → writes data/clip_demo.json
└── visualization/
    └── index.html            #   interactive dual-encoder + similarity matrix
```

## 1. Set up the environment

Requires Python 3.10+. From inside this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

This installs the CPU build of PyTorch — everything runs on a laptop with no GPU.

## 2. Run the demo

```bash
python demo/run_demo.py
```

Runs end-to-end on CPU in a few seconds. Expected output (numbers vary slightly
by machine but the story is identical):

```
CLIP demo | params=29,569 | 12 classes | chance=8.3%
step 700/700 | loss 1.49 | zero-shot acc  99.4%
Final zero-shot accuracy: 99.4%  (chance 8.3%)
mean diagonal (matched)   = 0.777
mean off-diagonal (mismatched) = -0.06
OK: zero-shot classification works far above chance.
```

It also writes `data/clip_demo.json` (similarity matrix + training curve) for the
visualization.

You can inspect the synthetic data generator on its own:

```bash
python data/shapes.py
```

## 3. Explore the visualization

Open `visualization/index.html` in any browser (works offline via `file://`).
It shows the dual-encoder architecture (click any block for its role and
equations), the contrastive objective, and — from a **real** demo run baked into
the page — the image↔text similarity matrix (bright diagonal) and the training
curve of loss vs. zero-shot accuracy.

To load fresh data instead of the baked-in sample, serve the folder:

```bash
python -m http.server 8000    # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper section | Component | File |
|---|---|---|
| §2 (Approach), Fig 3 | Dual encoder, shared space, learned temperature | `src/model.py` |
| §2.2 | Symmetric contrastive (InfoNCE) loss | `src/loss.py` |
| §2.4 | Image encoder (vision backbone) | `src/encoders.py` → `ImageEncoder` |
| §2.4 | Text encoder | `src/encoders.py` → `TextEncoder` |
| §3.1 | Zero-shot transfer via caption prompts | `demo/run_demo.py` |
| (data) | Synthetic (image, caption) pairs | `data/shapes.py` |
