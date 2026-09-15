# Mixtral of Experts

A faithful, minimal, CPU-only reproduction of the **Sparse Mixture-of-Experts**
transformer from *Mixtral of Experts* (Mistral AI, 2024 — arXiv
[2401.04088](https://arxiv.org/abs/2401.04088)). Everything needed to read, run,
and understand the idea lives in this folder: the paper PDF, a from-scratch
implementation, a runnable demo with data, and an interactive visualization.

Mixtral replaces the dense feed-forward block of a transformer with a **Sparse
MoE** block: a set of expert MLPs (8 in the paper) plus a small **router** that,
for every token, selects the **top-2** experts. Only those two run, so the model
has many parameters but a much smaller number of *active* parameters per token.

```
Mixtral of Experts/
├── mixtral_of_experts.pdf   # the paper
├── requirements.txt         # torch (CPU) + numpy
├── src/
│   ├── moe.py               # §2  the sparse MoE layer: top-2 routing + load-balance loss
│   ├── model.py             # §2  a tiny transformer whose FFN is the MoE block
│   └── data.py              # §5  multi-domain toy task (distinct sub-patterns)
├── data/
│   └── generate_data.py     # writes a few example sequences per domain
├── demo/
│   └── run_demo.py          # trains the MoE, reports routing / specialization / balance
└── visualization/
    └── index.html           # interactive: top-2 gate, param donut, token→expert heatmap
```

## What the demo demonstrates

1. **Sparse top-2 routing** — a router softmax over 8 experts, top-2 selection,
   gate-weight renormalization, and true **sparse dispatch** (each expert only
   processes its assigned tokens).
2. **Total vs active parameters** — with `top-2 of 8`, only a fraction of the
   parameters run per token.
3. **Load-balancing loss** — the Switch/Shazeer auxiliary loss; the demo trains
   with and without it and compares expert usage.
4. **Expert specialization** — after training on a 4-domain task, the router
   learns a stable, near-deterministic **token → expert** map.
5. **A worked top-2 gate** for one token.
6. **Contrast with Switch Transformers' top-1 routing.**

## 1. Set up the environment

Requires Python 3.10+. From inside this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## 2. Peek at the data (optional)

```bash
python data/generate_data.py
```

Writes `data/samples.json` with one example sequence per domain. Each domain uses
its own disjoint block of token ids and its own arithmetic step, so the domains
are clearly separable sub-patterns.

## 3. Run the demo

```bash
python demo/run_demo.py
```

Runs end-to-end on CPU in under 10 seconds and writes `data/mixtral_results.json`
for the visualization.

### Expected output (highlights)

```
experts               : 8   top-k routing: 2
TOTAL parameters       : 120,896
ACTIVE params / token  : 47,168  (only 2/8 experts run)   -> 39% active per token
[contrast] Switch top-1 would activate 34,880 params/token (1/8 experts)

final next-token accuracy : 100.0%
mean dominant-expert share per token: 96.5% (crisp specialization)

WITH aux   — balance 0.998, max/min usage 1.33x
WITHOUT aux— balance 0.982, max/min usage 2.60x
```

The router learns a **stable token→expert partition** (each token routes to one
dominant expert ~96% of the time), the model reaches 100% accuracy, and the
load-balancing loss tightens expert usage (max/min usage 2.60× → 1.33×). Numbers
are deterministic given the fixed seeds.

Note on load balancing: this toy task is *balanced*, so routing stays fairly even
even without the aux loss — the loss still visibly tightens it, and at real scale
it is what prevents a few experts from monopolizing all tokens ("expert
collapse").

## 4. Explore the visualization

Open `visualization/index.html` in any browser (works offline via `file://`).
It shows the top-2 gate for a real token, a total-vs-active parameter donut, the
token→expert specialization heatmap, and the with/without load-balancing usage
bars.

To load live data instead of the baked-in snapshot, serve the folder:

```bash
python -m http.server 8000    # then open http://localhost:8000/visualization/
```

## Top-2 (Mixtral) vs top-1 (Switch)

Both are sparse MoE with a router and a load-balancing loss. **Switch
Transformers** route each token to a *single* expert (top-1) for maximum
sparsity; **Mixtral** routes to *two* (top-2), combining their outputs for a
little more capacity per token at a modest extra compute cost. The demo reports
the active-parameter count under both.

## Code ↔ paper map

| Paper idea | Component | File |
|---|---|---|
| Sparse MoE FFN block | `MoELayer` (router + experts) | `src/moe.py` |
| Top-2 routing + gate renormalization | `probs.topk(2)`, renormalize | `src/moe.py` |
| Sparse dispatch (only chosen experts run) | per-expert gather / `index_add_` | `src/moe.py` |
| Load-balancing auxiliary loss | `aux = N · Σ f_i · P_i` | `src/moe.py` |
| Expert = SwiGLU MLP | `Expert` | `src/moe.py` |
| MoE inside a transformer | `MoETransformer` | `src/model.py` |
| Total vs active parameters | `active_params_per_token` | `src/model.py` |
| Expert specialization | token→expert routing map | `demo/run_demo.py` |
