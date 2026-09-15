# Native Sparse Attention (NSA)

A faithful, minimal, CPU-only reproduction of the core mechanism of
**"Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse
Attention"**, Yuan, Gao, Dai, Luo, Zhao, Zhang, Wu, Yu, Li, Yang, Huang, Wang,
Zhang, Wang, Zhou, Zeng, Ruan, Wang, Liang, et al. — 2025,
[arXiv:2502.11089](https://arxiv.org/abs/2502.11089). Everything needed to read,
run, and understand the paper's central idea lives in this folder: the paper
PDF, a from-scratch implementation, a runnable demo with data, and an
interactive visualization.

## Plain-English summary

Full attention lets every query look at **every** past key. That is `O(L²)` and
becomes the bottleneck for long contexts. Many sparse-attention schemes cut this
cost, but only *at inference* — they are bolted on after training, or use
non-differentiable index tricks that block gradients, so the model never learns
to use sparsity well.

**NSA is sparse *by construction* and *trainable end-to-end*.** Every query is
answered by **three parallel branches**, each scoring only a small subset of keys,
and their outputs are merged by a learned per-query **gate**:

- **Compression** (§3.3.1) — keys/values are mean-pooled blockwise into a small
  set of coarse tokens, giving cheap **global** context.
- **Selection** (§3.3.2) — the compression attention scores are reused as a
  block-importance signal to pick the **top-n** most relevant blocks; their
  fine-grained keys/values are attended to. Reusing the compression scores is
  what keeps selection **differentiable** — this is the natively-trainable
  sparse core.
- **Sliding window** (§3.3.3) — a small window of the most recent tokens, so the
  model never has to relearn **local** patterns through the sparse paths.

```
o = g_cmp · Attn(q, K̃_cmp, Ṽ_cmp)   +   g_slc · Attn(q, K_slc, V_slc)   +   g_win · Attn(q, K_win, V_win)
g = softmax(W_g · q)          (the three gate weights sum to 1)
```

The paper's headline result is that NSA **matches or beats full attention** on
language-modeling and reasoning benchmarks while being dramatically cheaper on
long contexts. This folder reproduces the *mechanism* at tiny scale.

## What the demo shows

The demo trains **two identical tiny decoder-only models head-to-head** — one
with full causal attention, one with NSA — on a long-range **repeat-induction**
task: a random block `base` of length `P` is repeated twice, so
`seq = base ++ base`. Predicting a token in the *second* copy requires attending
back one whole period (distance `P`) to the matching token in the first copy — a
long-range dependency the local sliding window (size `w < P`) cannot cover, so
the **selection** branch has to do real work. The demo then reports:

1. **Held-out accuracy** — NSA matches full attention (**~100% vs 100%** in the
   reference run) on second-copy next-token prediction.
2. **Positions saved** — how many key positions a full-context query touches:
   full attention scores all of them, NSA only its sparse budget
   (**~21% fewer** even at this tiny length of 63).
3. **Learned gate weights** — the three-branch mixture, showing the model leans
   on **selection** to make the long-range jump.
4. **Scaling curve** — NSA's per-query budget grows only sub-linearly, so at the
   paper's 64k-token regime it touches **~87% fewer** positions.

## Folder layout

```
Native Sparse Attention/
├── native_sparse_attention.pdf   # the paper
├── requirements.txt              # pinned deps (CPU PyTorch + numpy)
├── src/                          # from-scratch implementation
│   ├── nsa.py                    #   §3.3  NSA (compress + select + window + gate) & full-attn baseline
│   └── model.py                  #   tiny decoder-only model that uses either attention
├── data/
│   └── generate_data.py          #   toy repeat-induction sequence generator
├── demo/
│   └── run_demo.py               #   train full vs NSA head-to-head, compare (end to end)
└── visualization/
    └── index.html                #   interactive: 3-branch diagram, gates, scaling curve
```

## Setup

Requires Python 3.10+. Reuse the shared virtual environment, or create one:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

This installs the **CPU** build of PyTorch — no GPU needed.

## How to run

```bash
python data/generate_data.py     # (optional) writes data/sequences.json
python demo/run_demo.py          # trains full + NSA and compares (~50s on CPU)
```

The demo writes `data/nsa_run.json` (accuracy, positions, gate weights, scaling)
for the visualization.

## Expected output

```
Native Sparse Attention vs full attention — repeat-induction
seq_len=64 period=32 | NSA: block=8 select=3 window=8

Training FULL-attention model ...
Training NSA model ...
trained both in ~50s

Held-out second-half next-token accuracy:
  full attention: 100.0%
  NSA           : 100.0%

Key positions touched by a full-context query (context = 63):
  full attention:  63.0
  NSA           :  49.9   (21% fewer positions)

Learned NSA gate weights (avg over queries):
  compression: 0.27 | selection: 0.49 | window: 0.24
OK: NSA matches full-attention quality while touching fewer positions.
```

## Explore the visualization

Open `visualization/index.html` in any browser (works offline via `file://`).
It shows an interactive **three-branch diagram** (drag the query to see exactly
which key positions each branch touches), the two governing equations, the
**learned gate weights** and accuracy/position bars from this run, and a
**log-log scaling curve** contrasting NSA's per-query budget with full
attention's linear growth up to 64k tokens. To load fresh data instead of the
baked-in sample, serve the folder:

```bash
python -m http.server 8000    # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper section | Idea | File |
|---|---|---|
| §3.1, Eq. 5 | Three-branch gated attention | `src/nsa.py` → `NSAAttention` |
| §3.3.1 | Compression (blockwise K/V pooling) | `NSAAttention.forward` (cmp branch) |
| §3.3.2 | Top-n block selection via reused scores | `NSAAttention.forward` (slc branch) |
| §3.3.3 | Sliding window (local) | `NSAAttention.forward` (win branch) |
| baseline | Full causal attention | `src/nsa.py` → `FullAttention` |
| — | Tiny decoder, train/eval, comparison | `src/model.py`, `demo/run_demo.py` |

## Honest scope notes

This is a *mechanism* reproduction, not a scaled system. The real NSA is trained
on a 27B-parameter model over long (up to 64k-token) contexts, with a
**hardware-aligned kernel** (Triton/GPU, grouped-query, blockwise memory access)
that turns the sparsity into real wall-clock speedups. Here the three branches
are computed with plain **masked dense attention** (perfectly fine at the tiny
`L=64` used in the demo) and we report the *number of key positions each branch
would touch* to make the sparsity concrete, rather than a GPU FLOP/latency count.
The compression, differentiable top-n selection, sliding window, and the learned
gate are all implemented as described in §3.3. Because CPU sequences are short,
the measured position savings at `L=63` are modest (~20%); the baked scaling
curve extrapolates the exact learned configuration to show where the savings
become large (~87% at 64k), which is the regime the paper targets.
```
