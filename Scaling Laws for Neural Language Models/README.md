# Scaling Laws for Neural Language Models

A faithful, minimal, self-contained reproduction of the central empirical
finding of Kaplan et al., *"Scaling Laws for Neural Language Models"* (2020,
[arXiv:2001.08361](https://arxiv.org/abs/2001.08361)): the language-modeling
loss falls off as a smooth **power law** in the model's (non-embedding)
parameter count `N`.

Everything needed to read, run and understand the idea lives in this folder —
the paper PDF, a tiny-from-scratch decoder-only LM, a runnable demo, and an
interactive visualization.

## The core idea, in plain English

If you train language models of many different sizes on the same data and record
each model's final loss, the losses do not scatter randomly. They fall on a
straight line when you plot loss against model size on **log-log axes**. That
straight line is a power law:

```
L(N) ≈ (Nc / N) ^ alpha_N          (paper Eq. 1.1)
```

`alpha_N` is the *scaling exponent* (the slope of the log-log line) and `Nc` is a
constant. Kaplan et al. measure `alpha_N ≈ 0.076` for real language models over
seven orders of magnitude. The practical punchline: bigger models are
predictably better, and you can extrapolate the curve to decide how big to go.

## What the demo shows

`demo/run_demo.py` trains a **series of tiny decoder-only LMs of increasing
size** (varying width `d_model` and depth `n_layer`) on the *same* synthetic
corpus, records each model's final loss, and then fits the power law
`L(N) = (Nc/N)^alpha_N` by least squares in log-log space. It prints:

- a table of `(N, loss)` for every model,
- the fitted exponent `alpha_N`, the scale `Nc`, and the log-log `R²`,
- an assertion that the loss decreases as a clean straight line in log-log space.

It also writes `data/scaling_results.json`, which the visualization reads.

## Folder layout

```
Scaling Laws for Neural Language Models/
├── scaling_laws_for_neural_language_models.pdf   # the paper
├── requirements.txt              # CPU PyTorch + numpy
├── src/
│   ├── tiny_lm.py                # minimal decoder-only LM; counts non-embedding N (§2.1)
│   └── scaling.py                # log-log power-law fit for L(N) (§3)
├── data/
│   └── generate_data.py          # synthetic high-order-Markov corpus
├── demo/
│   └── run_demo.py               # train a size series, fit L(N), write JSON
└── visualization/
    └── index.html                # interactive log-log loss-vs-size plot
```

## Setup

Requires Python 3.10+. Reuse the shared environment from the reference folder,
or create your own:

```bash
source "../Attention Is All You Need/.venv/bin/activate"
# or, standalone:
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## How to run

```bash
python data/generate_data.py     # writes data/corpus.json (also auto-created by the demo)
python demo/run_demo.py          # trains the size series and fits the power law
```

## Expected output

The loss decreases monotonically with model size, and the log-log fit is tight
(`R² > 0.9`), e.g.:

```
 N (non-emb) | layers | d_model |  val loss
----------------------------------------------
       3,312 |      1 |      16 |    3.07
       7,272 |      1 |      24 |    2.98
      25,472 |      2 |      32 |    2.63
      56,640 |      2 |      48 |    2.29
     150,080 |      3 |      64 |    ...
     335,712 |      3 |      96 |    ...

Fitted power law  L(N) = (Nc / N) ^ alpha_N
  alpha_N (exponent) = ~0.07-0.12
  log-log R^2        = > 0.9
```

## Explore the visualization

Open `visualization/index.html` in any browser (works offline via `file://`).
It shows the fitted power law as an interactive log-log plot with the real
`(N, loss)` points from your demo run baked in, the scaling equation, and a
panel explaining why the relationship is a straight line in log-log space.

For live data, serve the folder so the page can fetch the generated JSON:

```bash
python -m http.server 8000    # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper section | Idea | File |
|---|---|---|
| §1, Eq. 1.1 | Power law `L(N) = (Nc/N)^alpha_N` | `src/scaling.py` |
| §2.1 | `N` counts non-embedding parameters | `src/tiny_lm.py` (`num_params`) |
| §3, Fig. 1 | Loss is a straight line in log-log space | `demo/run_demo.py`, `visualization/index.html` |
| §2.1 | Decoder-only Transformer LM | `src/tiny_lm.py` |
