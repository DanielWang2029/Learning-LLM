# GPT-4 — Predictable Scaling (a runnable, honest reproduction)

**Paper:** *GPT-4 Technical Report* — OpenAI, 2023 — [arXiv:2303.08774](https://arxiv.org/abs/2303.08774)

> **This is a closed model.** GPT-4's weights, architecture, dataset, and training
> recipe are **not public**, so no one can reproduce GPT-4 itself. This folder is
> honest about that. What it *does* reproduce is the single most **documented and
> verifiable** methodological contribution of the report:
> **predictable scaling** — the report states that GPT-4's final loss was
> predicted ahead of time from a family of much smaller models (up to ~10,000×
> less compute) using a power-law scaling curve. We reproduce exactly that
> workflow at tiny scale on a CPU, plus the report's practice of tracking
> **capability vs. scale**.

## The core idea, in plain English

Bigger language models get lower loss, and they do so in a strikingly regular
way: plot loss against model size on a log–log axis and you get (almost) a
straight line, bending toward an irreducible floor set by the data's own
entropy. That regularity is a **scaling law**:

```
L(N) = a · N^(-alpha) + E
```

If the law holds, you can fit it on cheap small models and **predict** the loss
of an expensive large model *before paying to train it*. The GPT-4 report says
OpenAI did precisely this. Capabilities (e.g. benchmark accuracy) tend to
improve with scale too, which the report tracks alongside loss.

## What the demo shows

`demo/run_demo.py`:

1. Builds a fixed synthetic language (a high-order Markov source) with a known
   irreducible entropy floor.
2. Trains a **family of five tiny GPTs** of increasing size, recording each
   one's validation loss and a 4-way **multiple-choice** accuracy.
3. Fits the power law `L(N)=a·N^-alpha+E` on the **four smaller** models only.
4. **Predicts** the loss of the largest, *held-out* model — then trains it and
   compares. In a sample run the prediction was **3.0015** vs. actual
   **2.9567** (a **1.5%** error).
5. Shows multiple-choice accuracy climbing from **~26% → ~44%** (chance = 25%),
   i.e. capability improving predictably with scale.

Everything runs on CPU in well under a minute.

## Folder layout

```
GPT-4/
├── gpt-4.pdf                 # the paper (unchanged)
├── requirements.txt          # torch==2.8.0, numpy>=1.26 (CPU)
├── src/
│   ├── model.py              # tiny decoder-only GPT (GPTConfig lets you dial size N)
│   ├── data.py               # fixed Markov language + its entropy floor
│   ├── scaling.py            # fit/extrapolate L(N)=a·N^-alpha+E (NumPy only)
│   └── eval.py               # multiple-choice capability harness
├── data/                     # generated: scaling_results.json (for the viz)
├── demo/run_demo.py          # end-to-end: train family → fit → predict → verify
└── visualization/index.html  # interactive scaling curve + capability-vs-scale
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

### Expected output (abridged)

```
model            params(N)    val loss    MC acc
------------------------------------------------
XS                   1,080      3.1427     25.8%
S                    3,696      3.1230     27.5%
M                   15,072      3.1044     34.2%
L                   26,240      3.0042     41.7%
XL (held out)       86,064      2.9567     44.2%

SCALING LAW  L(N) = a * N^(-alpha) + E ...
Extrapolation to the held-out XL model:
  predicted loss : 3.0015
  actual   loss : 2.9567
  relative error: 1.51%
OK: the loss of a larger model was predicted before training it.
```

(Exact numbers vary slightly by machine; the prediction error stays small.)

## Explore the visualization

Open `visualization/index.html` in any browser (works offline via `file://`).
It shows the **predictable-scaling curve** (fit models as dots, held-out model's
actual loss as a star landing on the extrapolated line) and a
**capability-vs-scale** chart. Real numbers from a demo run are baked in; serve
the folder over http to load live `data/scaling_results.json`:

```bash
python -m http.server 8000   # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Report topic | What we reproduce | File |
|---|---|---|
| "Predictable Scaling" — fit small, predict large | power-law fit + extrapolation to a held-out model | `src/scaling.py`, `demo/run_demo.py` |
| Loss scaling with model size | family of GPTs of increasing N; monotone loss ↓ | `src/model.py`, `demo/run_demo.py` |
| Irreducible loss term E | entropy floor of the synthetic source | `src/data.py` |
| Capability tracking across scale | multiple-choice accuracy vs. size | `src/eval.py` |
| Closed architecture/data | *not reproduced* — stated honestly above | — |
