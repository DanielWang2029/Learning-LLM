# Qwen2.5 Technical Report

A faithful, minimal, and fully self-contained reproduction of the **Qwen2.5**
architecture from Alibaba's *"Qwen2.5 Technical Report"* (2024,
arXiv:2412.15115). Everything needed to read, run, and understand the model
lives in **this folder**: the paper PDF, a from-scratch implementation, a
runnable demo, and an interactive visualization.

Qwen2.5 is a dense decoder-only Transformer. Most of its recipe (RMSNorm, RoPE,
Grouped-Query Attention, SwiGLU) is now standard, so this repo focuses on the two
choices that make Qwen *distinctive* against the otherwise Llama-style baseline:

- **QKV bias** — a bias term is added to the query/key/value projections (while
  the output projection stays bias-free), against the modern trend of dropping
  all biases; the authors report it helps length extrapolation.
- **Untied input/output embeddings** — the token embedding and the LM head are
  *separate* weight matrices (many small models tie them; Qwen2.5 spends the
  extra parameters on capacity).

```
Qwen2.5/
├── qwen25.pdf                  # the paper itself
├── requirements.txt            # pinned CPU dependencies (torch + numpy)
├── src/
│   ├── layers.py               #   RMSNorm, RoPE, SwiGLU (standard ingredients)
│   ├── attention.py            #   Grouped-Query Attention WITH QKV bias
│   └── model.py                #   full model + untied-embedding logic
├── data/
│   └── generate_demo_data.py   # writes a few example copy-task sequences
├── demo/
│   └── run_demo.py             # feature-check + training + ablation
└── visualization/
    └── index.html              # architecture diagram with feature callouts
```

## 1. Set up the environment

Requires Python 3.10+. From inside this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## 2. Run the demo

```bash
python data/generate_demo_data.py   # optional: writes data/samples.json
python demo/run_demo.py             # ~14s on CPU
```

The demo (1) builds the model and prints a **feature-check** confirming each
distinctive component is active, (2) trains it on a toy copy task, and (3) shows
post-training evidence the features are load-bearing plus a small QKV-bias
ablation.

## 3. Expected output

Every feature reports active, the copy task reaches **100% accuracy**, the QKV
bias learns a non-zero value, and the untied embeddings diverge from the LM head:

```
Feature-check (distinctive Qwen2.5 choices in bold):
  [x] QKV bias on Q/K/V projections
  [x] No bias on output projection
  [x] Untied input/output embeddings
  [x] RMSNorm pre-normalization
  [x] RoPE rotary positions
  [x] Grouped-Query Attention
  [x] SwiGLU feed-forward

Post-training evidence the features are load-bearing:
  learned QKV bias L2 norm      : 1.036  (was 0 at init)
  mean cos(embed_row, head_row) : -0.164  (=1.0 would mean tied)
```

Results are written to `data/qwen25_demo.json` for the visualization.

## 4. Explore the visualization

Open `visualization/index.html` in any browser for a color-coded map of the
Qwen2.5 block with callouts on the QKV bias and untied embeddings, a live
feature-check panel, and the training loss/accuracy curve — all using the real
numbers from your demo run. Works from `file://`; serve over HTTP to load fresh
JSON:

```bash
python -m http.server 8000     # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper topic | Component | File |
|---|---|---|
| Architecture — QKV bias | GQA with bias on Q/K/V | `src/attention.py` |
| Architecture — untied embeddings | separate LM head weight | `src/model.py` |
| Architecture — RMSNorm | pre-normalization | `src/layers.py` |
| Architecture — RoPE | rotary positions | `src/layers.py` |
| Architecture — SwiGLU | gated feed-forward | `src/layers.py` |
| Architecture — GQA | grouped K/V heads | `src/attention.py` |
