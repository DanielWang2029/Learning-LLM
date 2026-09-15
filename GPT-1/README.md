# GPT-1

A faithful, minimal, CPU-friendly reproduction of **GPT-1** — *"Improving
Language Understanding by Generative Pre-Training"* (Radford, Narasimhan,
Salimans & Sutskever, 2018). Everything needed to read, run and understand the
core idea lives in this folder.

## The idea in two paragraphs

GPT-1 introduced the now-standard recipe of **generative pre-training followed
by discriminative fine-tuning**. First, a decoder-only Transformer is trained
on lots of unlabeled text with a single self-supervised objective: predict the
next token. Because it is a decoder, a **causal mask** ensures each position
only sees the tokens before it, so the model learns a genuine left-to-right
language model.

The payoff comes in stage two. Instead of designing a bespoke model per task,
you take the pre-trained network, attach one small linear head to the final
token's representation, and fine-tune on a labeled task. The features learned
during pre-training transfer, so the model reaches high accuracy from far fewer
labels than a network trained from scratch.

## What the demo shows

`demo/run_demo.py` runs both stages on a toy char-level corpus of two-topic
sentences:

1. **Pre-training** — trains the LM to predict the next character; prints the
   **perplexity dropping** (~28 → ~1.3) and generates a coherent continuation
   of a prompt.
2. **Fine-tuning** — attaches a classifier head and fine-tunes on a *small*
   labeled topic-classification set, once from the pre-trained weights and once
   from scratch. The pre-trained model reaches **100%** vs the scratch model's
   ~83% from the same handful of labels — the transfer gain.

## Folder layout

```
GPT-1/
├── gpt-1.pdf               # the paper
├── requirements.txt        # torch (CPU) + numpy
├── src/                    # the implementation
│   ├── attention.py        #   causal (masked) multi-head self-attention
│   └── model.py            #   GPT decoder, LM head, fine-tuning classifier
├── data/
│   └── generate_data.py    #   writes a tiny two-topic char corpus
├── demo/
│   └── run_demo.py         #   pre-train + fine-tune, end to end
└── visualization/
    └── index.html          # interactive, offline architecture + charts
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## Run

```bash
python data/generate_data.py     # optional: writes data/corpus.json to inspect
python demo/run_demo.py          # ~10-20s on CPU
```

### Expected output

```
Pre-training done in ...s | perplexity 27.9 -> 1.32
Greedy continuation of 'the ':
  'the pine grows today . the pine grows today '
  [pretrained]   fine-tune 60 steps -> test accuracy 100.0%
  [from-scratch] fine-tune 60 steps -> test accuracy 83.0%
Transfer gain: pre-trained 100.0%  vs  scratch 83.0%  (+17.0 pts)
OK: pre-training lowered perplexity and transferred to the downstream task.
```

It also writes `data/gpt1_sample.json` (loss curve, transfer numbers, a causal
attention matrix) for the visualization.

## Explore the visualization

Open `visualization/index.html` in any browser (offline via `file://`). Click
blocks of the decoder to see equations, watch the perplexity curve, compare the
pre-trained vs from-scratch fine-tuning bars, read the generated sample, and
inspect the lower-triangular **causal** attention heatmap. For live data:
`python -m http.server 8000` → `http://localhost:8000/visualization/`.

## Code ↔ paper map

| Paper concept | Component | File |
|---|---|---|
| Decoder-only Transformer LM (§3.1) | `GPT`, pre-LN blocks | `src/model.py` |
| Masked self-attention / causal mask | `CausalSelfAttention` | `src/attention.py` |
| Generative pre-training objective (§3.1) | next-token `lm` loss | `src/model.py`, `demo/run_demo.py` |
| Task-specific fine-tuning (§3.2, §3.3) | `GPTClassifier` on last token | `src/model.py` |
| Transfer beats from-scratch | pretrained vs scratch comparison | `demo/run_demo.py` |
