# Gemma 3 — 5:1 Local:Global Attention

A faithful, minimal, self-contained reproduction of the signature efficiency idea
of **Gemma 3** (Google, 2025), arXiv:2503.19786. Everything needed to read, run,
and understand the idea lives in **this folder**: the paper PDF, a from-scratch
implementation, a runnable CPU demo, and an interactive visualization.

## Plain-English summary

Full ("global") self-attention is expensive: every token attends to every other,
so cost grows with the square of the sequence length. Gemma 3's fix is to make
**most layers local** — each token attends only within a sliding window — and only
a **few layers global**, in a **5:1 ratio** (five local layers per global layer).
The rare global layers still relay long-range information across the whole
sequence, so quality holds while attention memory drops sharply. Gemma 3 also uses
**QK-norm**: query and key vectors are normalized before the attention dot product.

## What this demo does

We train three tiny transformers (identical except for their attention pattern) on
a **long-context retrieval** task: a length-64 sequence of filler tokens hides a
`MARK` followed by a payload value; a `QUERY` at the end must recover that value —
which means routing information from a possibly distant position to the end.

- **5:1 local:global** — Gemma 3's pattern (`[local×5, global]`);
- **all-global** — full attention everywhere (accurate but expensive);
- **all-local** — sliding window everywhere (cheap but can't route long-range).

We verify the layer pattern, then report each model's accuracy (overall, and split
by near vs far payloads) and its **attention-memory footprint** = the number of
allowed attention-score entries summed over layers.

## What the demo shows

```
5:1 layer pattern (6 layers): ['local', 'local', 'local', 'local', 'local', 'global']
  -> 5 local : 1 global (ratio 5:1), QK-norm on

  5:1 local:global   | acc 100.0% (near 100%, far 100%) | attn entries   6876
  all-global         | acc 100.0% (near 100%, far 100%) | attn entries  24576
  all-local          | acc  25.8% (near  39%, far  13%) | attn entries   3336

5:1 uses 28% of all-global attention memory (72% saved), at 100% accuracy vs all-global 100%.

Example (payload at position 30, 33 tokens from query):
  ...['f6', 'MARK', 'VAL=0']... QUERY@63
  true value = 0, 5:1 model predicted = 0 OK
```

The 5:1 model matches all-global accuracy (including on **far** payloads) while
using **28%** of its attention memory. All-local is even cheaper but collapses to
**13%** on far payloads — it simply cannot move information across the sequence.
Runs in ~50s on CPU. Results are written to `data/gemma3_results.json` for the viz.

## Folder layout

```
Gemma 3/
├── gemma_3.pdf               # the paper
├── requirements.txt          # pinned deps (CPU PyTorch)
├── src/
│   ├── attention.py          # local/global attention masks + QK-norm
│   ├── model.py              # tiny transformer + 5:1 layer_pattern + memory proxy
│   └── task.py               # long-context retrieval ("needle") task
├── data/
│   └── generate_data.py      # writes the layer pattern + sample sequences
├── demo/
│   └── run_demo.py           # trains 5:1 vs all-global vs all-local, compares
└── visualization/
    └── index.html            # 5:1 layer stack + memory-vs-accuracy comparison
```

## Setup

Requires Python 3.10+. Reuse the shared virtual environment:

```bash
source "../Attention Is All You Need/.venv/bin/activate"
```

Or create a fresh one:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## How to run

```bash
python data/generate_data.py    # (optional) write data/sample_dataset.json
python demo/run_demo.py         # train 3 configs + compare, ~50s on CPU
```

Deterministic (`--seed`). Try `--seq-len`, `--window`, `--n-layers`, `--steps`.

## Explore the visualization

Open `visualization/index.html` in any browser (offline via `file://`). It shows
the 5:1 local/global layer stack (with each layer's attention footprint), and a
side-by-side accuracy-vs-memory comparison of the three patterns. Ships with
baked-in data; served over HTTP it reloads live results:

```bash
python -m http.server 8000   # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper idea | Where it lives | File |
|---|---|---|
| Sliding-window (local) attention | banded attention mask | `src/attention.py` → `attention_mask("local")` |
| Global attention | full attention mask | `src/attention.py` → `attention_mask("global")` |
| 5:1 local:global layer ratio | interleaved layer pattern | `src/model.py` → `layer_pattern(ratio=5)` |
| QK-norm | normalize q,k before dot product | `src/attention.py` → `Attention.forward` |
| Long-context efficiency | memory footprint per pattern | `src/model.py` → `attention_entries`, `demo/run_demo.py` |
