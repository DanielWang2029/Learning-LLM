# BERT

A faithful, minimal, CPU-friendly reproduction of **BERT** — *"BERT: Pre-training
of Deep Bidirectional Transformers for Language Understanding"* (Devlin, Chang,
Lee & Toutanova, 2018; [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)).
Everything needed to read, run and understand the core idea lives in this
folder: the paper PDF, a from-scratch implementation, a runnable demo with
data, and an interactive visualization.

## The idea in two paragraphs

Language models before BERT were mostly *left-to-right*: to predict a word they
could only look at the words before it. BERT is a Transformer **encoder**, so
every token attends to the **whole sentence at once** — both left and right
context. This bidirectionality is far more informative, but it breaks the usual
"predict the next word" objective (the answer would be visible).

BERT's fix is the **masked language model (MLM)** objective: randomly replace
~15% of the input tokens with a special `[MASK]` token and train the model to
recover the originals from the surrounding context. A `[CLS]` token is prepended
to every sequence; after pre-training, its final representation is used as a
sentence embedding for downstream **fine-tuning**. This repo implements the
encoder and the MLM head, and trains them on a tiny synthetic language.

## What the demo shows

`demo/run_demo.py` builds a toy "successor-grammar" corpus (each token has a
fixed successor, so a masked token is recoverable from its neighbours), then
trains the encoder with the MLM objective. It prints the **masked-token
accuracy climbing from chance (~4%) to ~99%**, shows a few worked
mask-and-predict examples, and captures a real self-attention heatmap for the
visualization.

## Folder layout

```
BERT/
├── bert.pdf                # the paper
├── requirements.txt        # torch (CPU) + numpy
├── src/                    # the implementation (the paper, in code)
│   ├── attention.py        #   bidirectional multi-head self-attention
│   ├── feedforward.py      #   position-wise feed-forward (GELU)
│   ├── layers.py           #   encoder layer & stack
│   └── model.py            #   embeddings, encoder, masked-LM head, pooler
├── data/                   # generated corpus + captured attention (gitignored)
│   └── generate_data.py    #   writes a tiny synthetic corpus
├── demo/
│   └── run_demo.py         #   masked language modelling, end to end
└── visualization/
    └── index.html          # interactive, offline architecture + attention viz
```

## Setup

Requires Python 3.10+. From inside this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

This installs the CPU build of PyTorch, so everything runs without a GPU.

## Run

```bash
python data/generate_data.py     # optional: writes data/corpus.json to inspect
python demo/run_demo.py          # trains the MLM; ~10-40s on CPU
```

### Expected output

```
Final masked-token accuracy: 98.7%
Example masked-LM predictions ([M] = masked position):
  input : 1 [M] 10 13 6 26 8 23 11 12 4
  truth : 25   pred : 25
...
OK: bidirectional encoder learned to fill in masked tokens.
```

It also writes `data/mlm_sample.json` (a real self-attention matrix) which the
visualization loads.

## Explore the visualization

Open `visualization/index.html` in any browser (works offline via `file://`).
Click any block of the encoder to see its role and equations, watch the
masked-LM objective on a real example, and inspect the trained bidirectional
self-attention heatmap. To load live (rather than baked-in) data, serve the
folder: `python -m http.server 8000` then open
`http://localhost:8000/visualization/`.

## Code ↔ paper map

| Paper concept | Component | File |
|---|---|---|
| Bidirectional self-attention (§3.1) | Multi-head self-attention, no causal mask | `src/attention.py` |
| Transformer encoder (§2, §3.1) | Encoder layer & stack (post-norm) | `src/layers.py` |
| Position-wise FFN with GELU | Feed-forward network | `src/feedforward.py` |
| Token / position / segment embeddings (§3) | `BertEmbeddings` | `src/model.py` |
| Masked LM objective (§3.1, Task #1) | `MaskedLMHead`, 15% masking (80/10/10) | `src/model.py`, `demo/run_demo.py` |
| `[CLS]` / `[MASK]` special tokens | reserved ids + pooler | `src/model.py` |
| Pre-train → fine-tune (§3.2) | `pooled_cls` for downstream tasks | `src/model.py` |
