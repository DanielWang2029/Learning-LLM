# Training Compute-Optimal LLMs — "Chinchilla"

A faithful, minimal, self-contained reproduction of the central idea of
Hoffmann et al., *"Training Compute-Optimal Large Language Models"* (2022,
[arXiv:2203.15556](https://arxiv.org/abs/2203.15556)), the paper behind the
**Chinchilla** model: given a fixed training-compute budget, there is an optimal
way to split it between **model size** and **training data**, and it sits at an
interior point — you should scale `N` and `D` together.

## The core idea, in plain English

Training compute is approximately

```
C ≈ 6 · N · D          (paper §2, Eq. 1)
```

where `N` is the number of model parameters and `D` is the number of training
tokens. If your compute `C` is fixed, then choosing a bigger model (`N`) forces
you to train on fewer tokens (`D = C / 6N`), and vice-versa.

- Make the model **too big** and it is *under-trained* (not enough tokens/steps).
- Make it **too small** and it is *capacity-limited* (it can't represent the data).
- The best loss is achieved by a model **in the middle** — and Chinchilla's
  headline result is that `N` and `D` should grow roughly *equally* as you scale
  compute. This implied that prior large models were oversized and under-trained.

## What the demo shows

`demo/run_demo.py` holds the compute budget `C` fixed and sweeps the split: for
each candidate model size `N` it sets the number of training tokens to
`D = C / (6N)` (so bigger models train for fewer steps), trains each tiny LM,
and records the final loss. The result is a **U-shaped curve** of loss vs. model
size with a clear interior minimum — the compute-optimal allocation. It prints a
table where the `6ND` column is (nearly) constant across all rows, confirming
every run used the same compute.

It writes `data/chinchilla_results.json`, which the visualization reads.

## Folder layout

```
Chinchilla/
├── chinchilla.pdf                # the paper
├── requirements.txt              # CPU PyTorch + numpy
├── src/
│   └── tiny_lm.py                # minimal decoder-only LM; N = non-embedding params
├── data/
│   └── generate_data.py          # synthetic high-order-Markov corpus
├── demo/
│   └── run_demo.py               # fixed-compute sweep over the N/D split
└── visualization/
    └── index.html                # interactive U-shaped compute-optimal frontier
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
python data/generate_data.py     # writes data/corpus.json (also auto-created by the demo)
python demo/run_demo.py          # sweeps allocations at fixed compute (~50s on CPU)
```

## Expected output

A U-shaped loss curve whose minimum is neither the smallest nor the largest
model, e.g.:

```
 N (params) |  steps |  D (tokens) |     6ND (C) |  val loss
      3,312 |   9000 |   9,216,000 |    1.83e+11 |    2.489   (capacity-limited)
      7,272 |   4099 |   4,197,376 |    1.83e+11 |    2.235   <- compute-optimal
     25,472 |   1170 |   1,198,080 |    1.83e+11 |    2.287
     56,640 |    526 |     538,624 |    1.83e+11 |    2.715
    150,080 |    199 |     203,776 |    1.83e+11 |    3.017
    335,712 |     89 |      91,136 |    1.84e+11 |    3.079   (under-trained)
```

Note how the `6ND` (compute) column stays constant — the trade-off is *at fixed
compute*.

## Explore the visualization

Open `visualization/index.html` in any browser (offline via `file://`). It plots
the real U-shaped frontier with the compute-optimal point highlighted, shows the
`C = 6ND` constraint, and includes the full sweep table (baked in from your demo
run). Serve the folder over HTTP to load live JSON instead:

```bash
python -m http.server 8000    # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper section | Idea | File |
|---|---|---|
| §2, Eq. 1 | Compute proxy `C ≈ 6ND` | `demo/run_demo.py` |
| §3 | Fixed-compute trade-off, interior optimum | `demo/run_demo.py`, `visualization/index.html` |
| §3 | Scale `N` and `D` together | `demo/run_demo.py` (sweep + best point) |
| — | Decoder-only Transformer LM, `N` = non-embedding params | `src/tiny_lm.py` |
