# Switch Transformers

A faithful, minimal, self-contained reproduction of the **Switch layer** from
Fedus et al., *"Switch Transformers: Scaling to Trillion Parameter Models with
Simple and Efficient Sparsity"* (2021,
[arXiv:2101.03961](https://arxiv.org/abs/2101.03961)): replace a transformer's
dense feed-forward sub-layer with a sparse **top-1 Mixture-of-Experts** and keep
the experts evenly used with a **load-balancing loss**.

## The core idea, in plain English

A normal transformer runs the same feed-forward network (FFN) on every token.
A Switch layer instead has **many** FFNs ("experts") and a small **router** that,
for each token, picks the single best expert (top-1) and runs only that one.

- **Parameters** grow with the number of experts (lots of capacity)…
- …but the **compute per token** stays constant (only one expert runs).

Left alone, a router tends to collapse — sending most tokens to a few favorite
experts. Switch adds a differentiable **load-balancing auxiliary loss**
(paper §2.2, Eq. 4-6):

```
aux = α · E · Σ_i  f_i · P_i
  f_i = fraction of tokens routed to expert i
  P_i = mean router probability for expert i
```

which is minimized when both are uniform, pushing the router to spread tokens
evenly across all `E` experts.

## What the demo shows

`demo/run_demo.py` trains a tiny transformer whose FFN sub-layer is a top-1
Switch MoE (implemented from scratch) and shows three things:

1. **Specialization.** On a "typed token" task where the label depends on the
   token's *type* and *value*, the router learns to send each type to its own
   expert. The per-type routing histogram becomes a clean diagonal (each type
   → one expert, 100% in the demo). To make this sharp in a tiny model, the
   experts only see a token's *value* while the router sees its *type*, so an
   expert given mixed types would face conflicting labels — specialization is
   the only low-loss solution.

2. **Load balancing.** On a task where any expert can serve any token, turning
   the auxiliary loss **off** lets the router collapse (one expert takes ~50% of
   tokens, another goes idle); turning it **on** restores an even ~25%-each split.

3. **Params vs. compute.** It reports that all experts together hold `~16.8k`
   parameters, but top-1 routing runs only `~4.2k` per token.

Results are written to `data/switch_results.json` for the visualization.

## Folder layout

```
Switch Transformers/
├── switch_transformers.pdf       # the paper
├── requirements.txt              # CPU PyTorch + numpy
├── src/
│   ├── moe.py                    # SwitchFFN: router, experts, top-1 dispatch, load-balancing loss (§2)
│   └── model.py                  # tiny transformer using the Switch FFN sub-layer
├── data/
│   └── generate_data.py          # the "typed token" task (type g, value v)
├── demo/
│   └── run_demo.py               # specialization + load-balancing experiments
└── visualization/
    └── index.html                # routing diagram + per-type histogram + load-balancing bars
```

## Setup

```bash
source "../Attention Is All You Need/.venv/bin/activate"
# or standalone:
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## How to run

```bash
python data/generate_data.py     # writes a sample typed-token task to data/
python demo/run_demo.py          # both experiments; ~5s on CPU
```

## Expected output

```
[A] SPECIALIZATION  (label depends on type AND value)
    task accuracy = 100.0%
    Per-type routing histogram (rows=type, cols=expert):
            exp0  exp1  exp2  exp3
    type0     0%    0%    0%  100%
    type1   100%    0%    0%    0%
    type2     0%    0%  100%    0%
    type3     0%  100%    0%    0%

[B] LOAD BALANCING  (label depends on value only)
    aux OFF: usage=[50% 0% 25% 25%]  max load=50%
    aux ON : usage=[25% 25% 25% 25%]  max load=25%

[C] PARAMS vs COMPUTE
    all 4 experts = 16,768 params, but top-1 routing runs only 1 = 4,192 params/token.
```

## Explore the visualization

Open `visualization/index.html` in any browser (offline via `file://`): a sparse
routing diagram (tokens → router → one expert each), the real per-type routing
histogram (diagonal = specialization), and the with/without load-balancing usage
bars. Serve over HTTP to load live JSON: `python -m http.server 8000`.

## Code ↔ paper map

| Paper section | Idea | File |
|---|---|---|
| §2.1 | Top-1 (Switch) routing, gated expert output | `src/moe.py` (`SwitchFFN`) |
| §2.2, Eq. 4-6 | Load-balancing auxiliary loss | `src/moe.py` (`aux`) |
| §2.1 | Switch layer replaces the dense FFN | `src/model.py` |
| §2 | Params scale with experts, compute stays top-1 | `demo/run_demo.py` (part C) |
