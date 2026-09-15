# Flamingo — a Visual Language Model for Few-Shot Learning

A faithful, minimal, fully self-contained reproduction of **Flamingo** from
Alayrac et al., *"Flamingo: a Visual Language Model for Few-Shot Learning"*
(2022), arXiv:[2204.14198](https://arxiv.org/abs/2204.14198). Everything needed
to read, run, and understand the core mechanism lives in **this folder**: the
paper PDF, a from-scratch implementation, a runnable demo with synthetic data,
and an interactive visualization.

## The core idea, in plain English

Flamingo turns a **frozen**, pretrained language model into one that can *see* —
without changing a single LM weight. It does this by inserting new, trainable
**gated cross-attention** layers between the frozen LM blocks. Vision reaches the
LM in three steps:

1. a (frozen) **vision encoder** turns an image into a grid of feature tokens;
2. a **Perceiver Resampler** compresses that grid into a small, *fixed* number of
   "visual tokens"; and
3. **gated cross-attention** lets the LM's text tokens attend to those visual
   tokens.

The key trick (paper §2.2, Fig 4) is the **tanh gate**: each new sub-layer is
scaled by `tanh(α)` with `α` initialized to **0**. Since `tanh(0) = 0`, the new
layers start as the identity and the model is *exactly* the pretrained LM. During
training `α` grows, smoothly "opening" the gates so vision flows in without
destabilizing the frozen weights.

## What the demo shows

We can't download image-text data, so we synthesize a tiny visual
question-answering task: an image of a colored shape, the fixed question
*"what color is the shape ?"*, and the color as the answer. The answer is
impossible to know from the text alone, so the model must use the image.

`demo/run_demo.py` runs the full story on CPU in ~15 s:

1. **Pretrain the LM on text only.** Its answer accuracy is stuck at chance
   (~25%) — the color simply isn't in the text.
2. **Freeze the LM, wrap it in Flamingo.** With gates at 0 we *verify* the model
   is byte-for-byte the frozen LM (`max|Flamingo − LM|` logits `= 0`).
3. **Train only the visual modules** (vision encoder + resampler + gated layers;
   69% of params, LM frozen). Answer accuracy jumps to **100%** as the gates open
   from 0.
4. It captures the gate-opening curve and a real spatial attention map (the
   answer token's attention, traced back onto the image grid, lands on the shape).

## Folder layout

```
Flamingo/
├── flamingo.pdf              # the paper itself
├── requirements.txt          # pinned CPU dependencies (torch + numpy)
├── src/                      # the paper, in code
│   ├── attention.py          #   scaled dot-product attention + FFN (shared)
│   ├── vision.py             #   §2.1  vision encoder (image -> feature grid)
│   ├── resampler.py          #   §2.1  Perceiver Resampler (grid -> R tokens)
│   ├── lm.py                 #   §2.2  frozen decoder-only language model
│   ├── gated_xattn.py        #   §2.2  GATED XATTN-DENSE (tanh gate, the key idea)
│   └── model.py              #   §2    full Flamingo (freeze LM, interleave gates)
├── data/
│   └── scenes.py             #   numpy generator: colored-shape VQA + vocabulary
├── demo/
│   └── run_demo.py           #   pretrain LM -> freeze -> train vision -> writes JSON
└── visualization/
    └── index.html            #   interactive pipeline + gate curves + attention map
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

Expected output (numbers vary slightly by machine, but the story is identical):

```
[1] Pretraining the language model on text only...
    LM text-only answer accuracy: 25.2%  (chance 25%)
[2] Gates initialized to 0 → tanh(0)=0 → model is the pure LM.
    max|Flamingo − LM| logits = 0.00e+00  (≈0 ⇒ identical)
[3] Training vision + resampler + gated cross-attention (LM stays frozen)...
    step 500/500 | loss 0.077 | acc 100.0% | mean|attn-gate| 0.103
    Final VQA answer accuracy: 100.0%  (LM-only was 25.2%, chance 25%)
OK: frozen LM learned to answer image questions through opened gates.
```

It also writes `data/flamingo_demo.json` (gate/accuracy curves + a spatial
attention example) for the visualization.

You can inspect the synthetic data generator on its own:

```bash
python data/scenes.py
```

## 3. Explore the visualization

Open `visualization/index.html` in any browser (works offline via `file://`).
It shows the vision → resampler → gated-cross-attention → frozen-LM pipeline
(click any stage), the tanh gating equations, and — from a **real** demo run
baked into the page — the gates opening from 0, accuracy rising from chance to
100%, and the answer token's attention landing on the shape in a held-out image.

To load fresh data instead of the baked-in sample, serve the folder:

```bash
python -m http.server 8000    # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper section | Component | File |
|---|---|---|
| §2 (Approach), Fig 3 | Full model: freeze LM, interleave gated layers | `src/model.py` |
| §2.1 | Vision encoder (image → feature tokens) | `src/vision.py` |
| §2.1 | Perceiver Resampler (→ fixed visual tokens) | `src/resampler.py` |
| §2.2, Fig 4 | Gated cross-attention (tanh gate) | `src/gated_xattn.py` |
| §2.2 | Frozen pretrained language model | `src/lm.py` |
| (task) | Synthetic colored-shape VQA + vocabulary | `data/scenes.py` |
