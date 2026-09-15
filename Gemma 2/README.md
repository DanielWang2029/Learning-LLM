# Gemma 2

A faithful, minimal, CPU-only reproduction of the **distinctive architectural
choices** of *Gemma 2* (Google, 2024 — arXiv
[2408.00118](https://arxiv.org/abs/2408.00118)). Everything needed to read, run,
and understand these ideas lives in this folder: the paper PDF, a from-scratch
implementation, a runnable demo with data, and an interactive visualization.

Gemma 2 is a family of open weights LLMs. Rather than re-training a 9B/27B model,
this folder reproduces the four design decisions the paper highlights and shows
each one working at laptop scale:

1. **Alternating local & global attention** — layers interleave a *local*
   sliding-window mask with a *global* full-causal mask (paper §2, Table 1).
2. **Logit soft-capping** — attention scores and final logits are squashed with
   `cap · tanh(x / cap)` (attn cap = 50, final cap = 30) so no logit runs away.
3. **Grouped-query attention (GQA)** — many query heads share fewer key/value
   heads.
4. **RMSNorm, pre- *and* post-norm** — every sub-block is wrapped in an RMSNorm
   before *and* after (a Gemma 2 detail).

```
Gemma 2/
├── gemma_2.pdf              # the paper
├── requirements.txt        # torch (CPU) + numpy
├── src/                     # from-scratch implementation
│   ├── config.py            #   §2  all architectural hyper-parameters
│   ├── norm.py              #   §2  RMSNorm (scale stored as 1 + w)
│   ├── attention.py         #   §2  GQA + sliding-window/global masks + attn soft-cap + RoPE
│   └── model.py             #   §2  decoder blocks, GeGLU MLP, final logit soft-cap
├── data/
│   └── generate_data.py     #   tiny "repeat-the-prefix" (induction) dataset
├── demo/
│   └── run_demo.py          #   trains WITH vs WITHOUT soft-capping and compares
└── visualization/
    └── index.html           #   interactive diagram + masks + soft-cap curves (real data)
```

## 1. Set up the environment

Requires Python 3.10+. From inside this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

This project reuses the shared CPU PyTorch build; no GPU or internet is needed at
runtime.

## 2. Generate the demo data (optional — the demo does this itself)

```bash
python data/generate_data.py
```

Writes `data/repeat_dataset.json`: each sequence is a random prefix, a `SEP`
token, then an exact copy of the prefix. Reproducing the copied half needs
long-range (global) attention; the copy is deterministic, so a trained model
becomes very confident — which makes soft-capping's effect easy to see.

## 3. Run the demo

```bash
python demo/run_demo.py
```

Runs end-to-end on CPU in about 20 seconds. It:

- prints the **model summary** (params, GQA grouping, the per-layer
  local/global schedule) and the two **attention masks** as ASCII art;
- trains two **identical** tiny models — one **with** logit soft-capping, one
  **without** — on the repeat task and logs loss, copy accuracy, and the peak
  output-logit magnitude;
- writes `data/gemma2_results.json` for the visualization.

### Expected output

```
WITH    soft-cap: initial loss   28.31 | copy-acc  98.6% | peak |output logit|   29.97 (<= final cap 30)
WITHOUT soft-cap: initial loss   88.18 | copy-acc   6.6% | peak |output logit|  112.27 (unbounded)
```

The takeaway: soft-capping (a) keeps every output logit inside `[-30, 30]` (peak
≈ 30) versus ≈ 112 uncapped, and (b) tames the initial-loss spike (88 → 28), so
the capped model converges to ~99% while the uncapped one stalls near chance.
(Exact numbers are deterministic given the fixed seeds.)

## 4. Explore the visualization

Open `visualization/index.html` in any browser (works offline via `file://`).
It shows:

- a **clickable decoder block** — embedding, pre/post RMSNorm, GQA, GeGLU,
  final soft-cap — each with its equation and source file;
- the **alternating layer schedule** and the exact **local vs global masks**
  rendered as heatmaps;
- the **soft-cap transfer curve** `cap·tanh(x/cap)` for cap = 30 and 50 against
  the identity line;
- the **real training dynamics** from your demo run: max output-logit magnitude
  (capped stays under the cap line; uncapped spikes) and accuracy (only the
  capped run converges);
- a **GQA diagram** showing query heads sharing key/value heads.

To load live data instead of the baked-in snapshot, serve the folder:

```bash
python -m http.server 8000    # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper idea (§2 / Table 1) | Component | File |
|---|---|---|
| Alternating local/global attention | sliding-window + causal masks, per-layer schedule | `src/attention.py`, `src/model.py` |
| Sliding-window (local) attention | `build_sliding_window_mask` | `src/attention.py` |
| Grouped-query attention (GQA) | fewer KV heads, `repeat_interleave` | `src/attention.py` |
| Attention logit soft-capping (cap 50) | `cap·tanh(scores/cap)` before softmax | `src/attention.py` |
| Final logit soft-capping (cap 30) | `soft_cap()` on vocab logits | `src/model.py` |
| RMSNorm (pre + post norm) | `RMSNorm`, `DecoderBlock` | `src/norm.py`, `src/model.py` |
| GeGLU feed-forward | `GeGLU` | `src/model.py` |
| RoPE positional encoding | `_apply_rope` | `src/attention.py` |
| Tied embeddings / √d_model scaling | `Gemma2Model` | `src/model.py` |
