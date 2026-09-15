# LLaVA — Visual Instruction Tuning (a runnable reproduction)

**Paper:** *Visual Instruction Tuning* — Liu, Li, Wu, Lee, 2023 —
[arXiv:2304.08485](https://arxiv.org/abs/2304.08485)

A faithful, tiny reproduction of LLaVA's core recipe: connect a **frozen**
vision encoder to a language model through a single **learned projection**, then
**instruction-tune**. Everything runs on CPU in a few seconds.

## The core idea, in plain English

You already have a good vision model (CLIP) and a good language model. LLaVA's
insight is that you don't need to retrain either — you just need to **translate**
what the vision model sees into the language model's own "word" space. That
translator is a small **projection** matrix/MLP. Freeze the vision tower, keep
the LM mostly as-is, learn the projection (plus light tuning) on
image + instruction + answer triples, and the LM can suddenly answer questions
about images.

## What the demo shows

`demo/run_demo.py`:

1. **Generates** synthetic images with NumPy — each has one shape
   (circle/square/triangle) of one color (red/green/blue/yellow) — paired with
   an instruction ("what color?" / "what shape?") and the true answer.
2. **Freezes** a tiny CNN vision encoder (a stand-in for CLIP). It emits 9 patch
   tokens per image and is never trained.
3. **Learns only** the projection + a tiny LM by instruction tuning.
4. **Tests on held-out images** and reports accuracy — well above chance.
5. **Ablates the projection** (feeds the LM zeros instead of visual tokens): the
   model goes blind and drops to chance. It also trains a fully **blind** model
   for comparison.

### Result from a sample run

| condition | overall | color Q | shape Q |
|---|---|---|---|
| **FULL** (projection trained) | **96.0%** | 99.4% | 91.9% |
| ABLATED (projection → zeros) | 27.7% | 25.5% | 30.4% |
| BLIND (trained without vision) | 27.7% | — | — |

Chance is ≈ 25% (color, 1 of 4) / 33% (shape, 1 of 3). The learned projection —
not the frozen encoder, not the LM alone — is what lets the model see.

## Folder layout

```
LLaVA/
├── llava.pdf                 # the paper (unchanged)
├── requirements.txt          # torch==2.8.0, numpy>=1.26 (CPU)
├── src/
│   ├── data.py               # NumPy shape/color image generator + token vocab
│   ├── vision.py             # FROZEN tiny-CNN vision encoder (CLIP stand-in)
│   └── model.py              # learned Projector + tiny LM + LlavaTiny wrapper
├── data/                     # generated: llava_results.json (for the viz)
├── demo/run_demo.py          # generate → freeze → instruction-tune → test → ablate
└── visualization/index.html  # architecture + held-out image predictions + ablation
```

## Setup

Reuse the shared CPU environment (do **not** create a new venv):

```bash
source "/workspace/Attention Is All You Need/.venv/bin/activate"
```

Or, standalone:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## How to run

```bash
python demo/run_demo.py
```

Prints the accuracy table above plus concrete held-out predictions, and writes
`data/llava_results.json`.

## Explore the visualization

Open `visualization/index.html` in any browser (offline via `file://`). It draws
the vision → projection → LM pipeline, renders the **real held-out images** with
the model's answers (full vs. ablated), and charts accuracy by condition and
question type. Serve over http to load live `data/llava_results.json`:

```bash
python -m http.server 8000   # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper idea | What we reproduce | File |
|---|---|---|
| Frozen vision encoder (CLIP ViT) | frozen tiny CNN → patch tokens | `src/vision.py` |
| Projection connecting vision to LM | learned MLP projector (LLaVA-1.5 style) | `src/model.py` → `Projector` |
| Multimodal input `[visual tokens | text]` | concat projected patches + instruction | `src/model.py` → `LlavaTiny.forward` |
| Visual instruction tuning | train projector + LM on (image, instr, answer) | `demo/run_demo.py` |
| Projection is the bridge (ablation) | zero visual tokens → chance accuracy | `demo/run_demo.py` (`ablate_projection`) |
