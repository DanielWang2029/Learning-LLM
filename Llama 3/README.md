# The Llama 3 Herd of Models

A faithful, minimal, and fully self-contained reproduction of the **Llama 3**
recipe from Meta's *"The Llama 3 Herd of Models"* (2024, arXiv:2407.21783).
Everything needed to read, run, and understand the architecture lives in **this
folder**: the paper PDF, a from-scratch implementation, a runnable demo with
data, and an interactive visualization.

Llama 3 is a *dense* decoder-only Transformer. Its published architecture (§3.1)
is deliberately conventional — the gains come from scale, data, and a set of
well-chosen ingredients:

- **RMSNorm** pre-normalization
- **RoPE** rotary positions with the base frequency raised to **500,000**
- **Grouped-Query Attention (GQA)** — 8 K/V heads shared across the query heads
- **SwiGLU** feed-forward
- a much larger **128K byte-level BPE tokenizer** (up from Llama 2's 32K)

This repo implements all of these at tiny scale and demonstrates the two Llama 3
headlines you can actually measure on a laptop: (1) the recipe learns, and (2) a
larger vocabulary encodes the same text in **fewer tokens**.

```
Llama 3/
├── llama_3.pdf                 # the paper itself
├── requirements.txt            # pinned CPU dependencies (torch + numpy)
├── src/                        # the recipe, in code
│   ├── normalization.py        #   §3.1  RMSNorm pre-normalization
│   ├── rope.py                 #   §3.1  Rotary Position Embeddings (base 500k)
│   ├── attention.py            #   §3.1  Grouped-Query Attention + repeat_kv
│   ├── feedforward.py          #   §3.1  SwiGLU MLP
│   ├── block.py                #   §3.1  pre-norm transformer block
│   ├── model.py                #   §3.1  full Llama 3 decoder-only model
│   └── tokenizer.py            #   §3.1  from-scratch byte-level BPE tokenizer
├── data/
│   └── generate_demo_data.py   # writes the small text corpus for BPE training
├── demo/
│   └── run_demo.py             # trains the model + runs the tokenizer study
└── visualization/
    └── index.html              # interactive recipe diagram + tokenizer chart
```

## 1. Set up the environment

Requires Python 3.10+. From inside this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

This installs the CPU build of PyTorch, so everything runs on a laptop with no GPU.

## 2. Run the demo

```bash
python data/generate_demo_data.py   # writes data/corpus.txt
python demo/run_demo.py             # ~10s on CPU
```

The demo has two parts:

**Architecture.** Trains a tiny Llama 3 model on a toy sequence-copy task. It
prints a *feature-check* confirming each ingredient is active (including the GQA
KV-cache reduction factor `n_heads / n_kv_heads`) and the loss/accuracy curve.

**Tokenizer.** Trains byte-level BPE tokenizers at several vocabulary sizes on a
small English corpus and encodes a held-out sentence with each. As the vocabulary
grows, more subwords merge and the same text needs fewer tokens.

## 3. Expected output

The copy task reaches **100% exact-copy accuracy**, and the tokenizer table shows
token counts falling as the vocabulary grows:

```
Feature-check | RMSNorm: on | RoPE base: 500000 | SwiGLU: on | GQA: 4 query heads
              share 2 K/V heads  ->  KV-cache reduction x2

  vocab_size | tokens | bytes/token | vs 256-byte
         256 |    144 |       1.000 |       0.0%
         384 |     51 |       2.824 |      64.6%
         512 |     41 |       3.512 |      71.5%
         640 |     33 |       4.364 |      77.1%
```

Growing the vocabulary 256 → 640 cuts the held-out text from 144 to 33 tokens and
raises compression from 1.00 to ~4.4 bytes/token — the same lever Llama 3 pulls by
moving from a 32K to a 128K vocabulary. Results are written to
`data/llama3_demo.json` for the visualization.

## 4. Explore the visualization

Open `visualization/index.html` in any browser for a color-coded map of the
Llama 3 recipe (click any block for its role and equations), a Grouped-Query
Attention diagram, and a live bar chart of the tokenizer-efficiency study using
the real numbers from your demo run. The page works straight from `file://`
(data is baked in); serve the folder over HTTP to load fresh JSON:

```bash
python -m http.server 8000     # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper section | Component | File |
|---|---|---|
| §3.1 | RMSNorm pre-normalization | `src/normalization.py` |
| §3.1 | RoPE (base 500,000) | `src/rope.py` |
| §3.1 | Grouped-Query Attention | `src/attention.py` |
| §3.1 | SwiGLU feed-forward | `src/feedforward.py` |
| §3.1 | Decoder-only model | `src/model.py` |
| §3.1 | 128K byte-level BPE tokenizer | `src/tokenizer.py` |
