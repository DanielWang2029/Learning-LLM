# T5

A faithful, minimal, CPU-friendly reproduction of **T5** — *"Exploring the
Limits of Transfer Learning with a Unified Text-to-Text Transformer"* (Raffel,
Shazeer, Roberts, Lee, Narang, Matena, Zhou, Li & Liu, 2020;
[arXiv:1910.10683](https://arxiv.org/abs/1910.10683)). Everything needed to read,
run and understand the core idea lives in this folder.

## The idea in two paragraphs

T5's thesis is that **every** NLP problem can be cast as **text-to-text**: the
model reads an input string and produces an output string, whether the task is
translation, summarization, classification or regression. This lets a single
encoder–decoder Transformer, with a single training objective and loss, handle
many tasks — you just prefix the input with a description of the task.

T5 is pre-trained with a **span-corruption** objective: random contiguous spans
of the input are replaced by unique *sentinel* tokens, and the decoder learns to
regenerate the missing spans (each announced by its sentinel). After this
self-supervised pre-training, the same model is fine-tuned on downstream tasks,
all framed as text-to-text. This repo implements the encoder–decoder, the
span-corruption objective, and three toy tasks.

## What the demo shows

`demo/run_demo.py` runs both stages:

1. **Span-corruption pre-training** on toy sentences (a fixed successor grammar,
   so dropped spans are predictable); prints the reconstruction accuracy
   (~95%).
2. **Text-to-text multitask** training of the *same* model on three tasks framed
   identically as "input → output" and selected by a task token: **copy**,
   **reverse**, **sort**. Reports per-task exact-match accuracy via greedy
   decoding (copy 100%, reverse ~100%, sort ~95%).

## Folder layout

```
T5/
├── t5.pdf                  # the paper
├── requirements.txt        # torch (CPU) + numpy
├── src/                    # the implementation
│   ├── attention.py        #   scaled dot-product & multi-head attention
│   ├── layers.py           #   encoder / decoder layers & stacks
│   └── model.py            #   full encoder-decoder T5 + greedy decode
├── data/
│   └── generate_data.py    #   vocab, span corruption, the three tasks
├── demo/
│   └── run_demo.py         #   pre-train (span corruption) + multitask
└── visualization/
    └── index.html          # interactive, offline text-to-text + cross-attention
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## Run

```bash
python data/generate_data.py     # optional: prints a span-corruption + task example
python demo/run_demo.py          # ~1 minute on CPU
```

### Expected output

```
  span reconstruction token accuracy: 95.2%
== Exact-match accuracy per task (greedy decoding) ==
     copy: 100.0%   [27, 17, 30, 21, 18, 25] -> [27, 17, 30, 21, 18, 25] ...
  reverse: 100.0%   [17, 17, 14, 20, 24, 16] -> [16, 24, 20, 14, 17, 17] ...
     sort:  95.0%   [13, 13, 29, 26, 18, 13] -> [13, 13, 18, 26, 29 ...] ...
Mean task accuracy: 98.3%
OK: one text-to-text model pre-trained by span corruption solves all three tasks.
```

It writes `data/t5_sample.json` (per-task accuracy, span reconstruction, loss
curves, a real decoder→encoder cross-attention matrix) for the visualization.

## Explore the visualization

Open `visualization/index.html` in any browser (offline via `file://`). See the
text-to-text framing of each task, a worked span-corruption example, per-task
accuracy bars, and the decoder→encoder **cross-attention** heatmap (for
`reverse`, note the bright anti-diagonal). For live data:
`python -m http.server 8000` → `http://localhost:8000/visualization/`.

## Code ↔ paper map

| Paper concept | Component | File |
|---|---|---|
| Encoder–decoder Transformer (§2.1) | `T5`, `Encoder`, `Decoder` | `src/model.py`, `src/layers.py` |
| Self / cross / causal attention | `MultiHeadAttention` + masks | `src/attention.py`, `src/layers.py` |
| Span-corruption objective (§3.1.4) | `corrupt_spans`, sentinels | `data/generate_data.py` |
| Unified text-to-text framework (§2.4) | task tokens + `task_example` | `data/generate_data.py`, `demo/run_demo.py` |
| Multitask, one model | copy / reverse / sort eval | `demo/run_demo.py` |
