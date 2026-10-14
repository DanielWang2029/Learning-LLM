# Qwen2.5-Coder

**Qwen Team · 2024 · arXiv:2409.12186**

Qwen2.5-Coder continues a code-specialized language model with both ordinary
next-token prediction and **fill-in-the-middle (FIM)** prediction.  FIM moves
the right-hand code context before the missing span:
`<PRE> prefix <SUF> suffix <MID> middle <END>`.  A causal decoder can then use
both sides while generating the middle.

This folder is a tiny, CPU-only reproduction of that documented mechanism, not
a reproduction of the paper's models, data scale, or benchmark results.  It
trains two small decoder-only Transformers on a deterministic toy-code grammar:
a FIM decoder and a next-token-only control.  The suffix comment identifies the
operation required in the missing expression, making right context essential.

## What the demo shows

The demo prints exact-match infill accuracy for both training objectives and a
concrete prefix/suffix/middle completion.  The FIM decoder learns to emit the
two-token expression after `<MID>`; the baseline was never trained on the FIM
sentinels or reordered context and fails this infill interface.

## Folder layout

```text
Qwen2.5-Coder/
├── qwen25-coder.pdf          # supplied paper
├── requirements.txt          # CPU-only PyTorch and NumPy
├── src/fim_model.py          # minimal causal decoder and greedy infilling
├── data/generate_data.py     # structured synthetic code / FIM transform
├── demo/run_demo.py          # train, compare, and write results JSON
└── visualization/index.html  # offline interactive explanation
```

## Setup

Python 3.10+ is sufficient.  Install the CPU build:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## Run

```bash
python demo/run_demo.py
```

Expected output is a clear FIM advantage, typically `100.0%` FIM exact match
against `0.0%` for the next-token-only control, plus a correctly completed
`return x <op> <number>` expression.  The run writes
`data/fim_results.json`, which contains the real metric and example used by
the visualization's baked-in default.

Open `visualization/index.html` directly in a browser.  It works offline under
`file://`; serving this folder with `python -m http.server` optionally lets it
load the freshly generated JSON.

## Code ↔ paper map

| Paper section | Reproduced idea | File |
|---|---|---|
| §3.1.1, tokenizer | FIM special-token order | `data/generate_data.py` |
| §3.2.1 | Next-token and file-level FIM objectives | `demo/run_demo.py` |
| Figure 3 | `<PRE> prefix <SUF> suffix <MID> middle` | `data/generate_data.py` |
| §6.2 | Exact-match code infilling comparison | `demo/run_demo.py` |
