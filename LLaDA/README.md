# LLaDA — Large Language Diffusion Models

A faithful, minimal, CPU-only reproduction of the core mechanism of
**"Large Language Diffusion Models"** (LLaDA), Nie, Zhu, Dong, Zhang, Yang,
Nie, Wang, Zhou, Li — 2025, [arXiv:2502.09992](https://arxiv.org/abs/2502.09992).
Everything needed to read, run, and understand the paper's central idea lives
in this folder: the paper PDF, a from-scratch implementation, a runnable demo
with data, and an interactive visualization.

## Plain-English summary

Almost every large language model is **autoregressive**: it writes text strictly
left to right, one token at a time. LLaDA shows this is not the only way. It
trains a language model as a **masked diffusion process**, closely related to the
"noise → denoise" idea behind image diffusion, but where the corruption is
*masking* tokens rather than adding Gaussian noise:

- **Forward process** — pick a masking ratio `t ∈ (0, 1]` and independently
  replace each token with `[MASK]` with probability `t`. At `t = 1` the whole
  sequence is masked; as `t → 0` it is left intact.
- **Model** — a single **bidirectional** Transformer (no causal mask) predicts
  the original token at every masked position, reading context from *both* sides.
- **Training loss** — cross-entropy on the masked positions, weighted by `1/t`
  (a Monte-Carlo estimate of an upper bound on the negative log-likelihood,
  paper Eq. 3).
- **Reverse process (generation)** — start from an **all-`[MASK]`** sequence and
  iteratively denoise over `T` steps. At each step the model predicts every
  masked token, **commits** the most confident predictions, and re-masks the
  rest ("low-confidence remasking"). Generation is therefore
  **non-autoregressive**: tokens are filled in *confidence* order, not left to
  right.

LLaDA's headline result is that a diffusion LM trained this way is competitive
with strong autoregressive LLMs — evidence that autoregression is not the source
of LLM capabilities. This folder reproduces the *mechanism* at tiny scale.

## What the demo shows

The demo trains a small masked-diffusion LM on toy **periodic** sequences (a
random motif of length `P` tiled across the sequence, so `seq[i] == seq[i+P]`).
Periodicity is a hard *global* constraint that a non-autoregressive generator
must satisfy across the whole sequence at once. The demo then:

1. **Reconstructs** held-out sequences with 50% of tokens masked, filling every
   blank in a single forward pass (~90% masked-token accuracy).
2. **Generates** brand-new sequences from an all-`[MASK]` start via `T`-step
   iterative unmasking, and checks how many are valid periodic sequences
   (**100%** in the reference run).
3. Prints a **step-by-step unmasking trace** showing tokens being committed in
   confidence order — visibly *not* left to right.

## Folder layout

```
LLaDA/
├── llada.pdf                  # the paper
├── requirements.txt           # pinned deps (CPU PyTorch + numpy)
├── src/                       # from-scratch implementation
│   ├── model.py               #   §2.2  bidirectional Transformer mask predictor
│   └── diffusion.py           #   §2    forward masking, loss, reverse sampling
├── data/
│   └── generate_data.py       #   toy periodic-sequence generator
├── demo/
│   └── run_demo.py            #   train → reconstruct → generate (end to end)
└── visualization/
    └── index.html             #   interactive: diffusion vs AR + unmasking animation
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
python demo/run_demo.py          # trains + reconstructs + generates (~35s on CPU)
```

The demo writes `data/llada_run.json` (metrics + unmasking trace) for the
visualization.

## Expected output

```
Masked-diffusion LM | params=227,819 | vocab=11 seq_len=12 steps=12
  step 1500/1500 | diffusion loss 0.68
trained in ~33s

Reconstruction (50% masked, one-shot fill):
  masked-token accuracy :  ~90%
  exact-sequence accuracy:  ~86%

Generation from all-[MASK] (256 samples, 12 steps):
  valid periodic-sequence rate: 100.0%
  distinct sequences generated : 32/256

Step-by-step unmasking trace (one sample; '_' = still masked):
  step  1/12 [11 masked]:  _  _  _  _  _  _  _  _  _  4  _  _
  ...
  step 12/12 [ 0 masked]:  4  4  3  4  4  3  4  4  3  4  4  3
OK: masked diffusion reconstructs and generates valid structured sequences.
```

## Explore the visualization

Open `visualization/index.html` in any browser (works offline via `file://`).
It contrasts masked diffusion with autoregressive generation, shows the three
governing equations, and **animates the real unmasking trace** captured from the
demo (tokens light up as they are committed — in confidence order, not left to
right). To load fresh data instead of the baked-in sample, serve the folder:

```bash
python -m http.server 8000    # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper section | Idea | File |
|---|---|---|
| §2.1 | Forward masking process (ratio `t`) | `src/diffusion.py` → `forward_mask` |
| §2.2 | Bidirectional mask predictor | `src/model.py` → `MaskPredictor` |
| §2.2, Eq. 3 | `1/t`-weighted masked cross-entropy | `src/diffusion.py` → `diffusion_loss` |
| §2.4 | Reverse process, low-confidence remasking | `src/diffusion.py` → `generate` |
| §3 | Reconstruction & generation evaluation | `demo/run_demo.py` |

## Honest scope notes

This is a *mechanism* reproduction, not a scaled model. The real LLaDA is an 8B
model trained on 2.3T tokens; here the "language" is a toy periodic grammar and
the model is ~0.23M parameters so it trains in seconds on a laptop CPU. The
forward/reverse processes, the bidirectional predictor, the `1/t`-weighted loss,
and low-confidence remasking are implemented as described in the paper. For
numerical stability at this tiny scale we clamp the sampled ratio `t` away from
`0` and normalize the loss by sequence length; these choices are documented in
`src/diffusion.py`.
