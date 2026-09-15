# Mistral 7B — Sliding Window Attention, GQA & Rolling KV Cache

**Paper:** *Mistral 7B* — Jiang et al., 2023 —
[arXiv:2310.06825](https://arxiv.org/abs/2310.06825)

A faithful, minimal reproduction of the three efficiency mechanisms Mistral 7B
is built around, at tiny scale on CPU: **sliding window attention (SWA)**,
**grouped-query attention (GQA)**, and a **rolling-buffer KV cache**.

## The core idea, in plain English

Full self-attention lets every token look at every earlier token — accurate but
expensive: cost and KV-cache memory grow with the sequence length. Mistral makes
three moves:

1. **Sliding Window Attention** — each token attends only to the previous **W**
   tokens (a causal *band*, not a full triangle). Per-token cost is O(W). Yet
   because layers stack, information still flows far: after *k* layers the
   receptive field is ~*k·W* tokens.
2. **Grouped-Query Attention** — use fewer key/value heads than query heads.
   Each KV head is shared by a group of query heads, shrinking the KV cache by
   `n_head / n_kv_head`.
3. **Rolling-buffer KV cache** — since a token never attends beyond W steps
   back, the cache only needs the last **W** keys/values. New entries overwrite
   the oldest, so cache size is **constant** regardless of sequence length.

## What the demo shows

`demo/run_demo.py` proves all three, with real training and real numbers:

1. **SWA works, and the window must be big enough.** On a task where each token
   is `(prev + prev2) mod V` (solvable from the last 2 tokens), accuracy is:

   | window W | next-token accuracy |
   |---|---|
   | 1 | **8.4%** (can't see both needed tokens) |
   | 2 | **99.8%** (just enough) |
   | 4 | **100%** |

2. **The mask is banded** — it prints the sliding-window mask (a diagonal band).
3. **The rolling buffer is exact.** Decoding token-by-token with a rolling KV
   cache matches a full forward pass: `max |logit diff| = 8e-6`, 100% token
   agreement.
4. **Memory is bounded.** KV-cache size stays flat past `L = W` for SWA while
   full attention grows linearly — a **8192×** reduction at L=16384 here
   (including GQA's 2× KV shrink).

Runs on CPU in under 30 seconds.

## Folder layout

```
Mistral 7B/
├── mistral_7b.pdf            # the paper (unchanged)
├── requirements.txt          # torch==2.8.0, numpy>=1.26 (CPU)
├── src/
│   ├── attention.py          # SlidingWindowAttention + GQA + RollingKVCache + mask
│   └── model.py              # tiny Mistral-style LM (full & cached decode paths)
├── data/                     # generated: mistral_results.json (for the viz)
├── demo/run_demo.py          # train SWA -> mask -> cache==full -> memory curve
└── visualization/index.html  # banded mask + rolling buffer + memory chart + GQA
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

Prints the window/accuracy table, the banded mask, the cache-equivalence check,
and the memory curve; writes `data/mistral_results.json`.

## Explore the visualization

Open `visualization/index.html` in any browser (offline via `file://`). It shows
the **banded attention mask**, an interactive **rolling-buffer** stepper, the
**KV-cache memory** curve (SWA flat vs. full linear, log–log), the **window vs.
accuracy** bars, and a **GQA** head-sharing diagram. Serve over http to load live
data:

```bash
python -m http.server 8000   # then open http://localhost:8000/visualization/
```

> Note: to keep the focus on SWA/GQA/rolling cache, this mini-model uses learned
> absolute positional embeddings rather than Mistral's RoPE; the three headline
> mechanisms are implemented faithfully.

## Code ↔ paper map

| Paper mechanism | What we reproduce | File |
|---|---|---|
| Sliding Window Attention | banded causal mask, O(W) attention | `src/attention.py` → `sliding_window_mask`, `SlidingWindowAttention.forward` |
| Grouped-Query Attention | `n_kv_heads < n_heads`, KV shared per group | `src/attention.py` → `_expand_kv` |
| Rolling-buffer KV cache | fixed last-W cache; exact vs. full | `src/attention.py` → `RollingKVCache`; `src/model.py` → `forward_cached` |
| Bounded memory at long context | constant cache past L=W | `demo/run_demo.py` (memory curve) |
