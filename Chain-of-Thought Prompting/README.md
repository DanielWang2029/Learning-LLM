# Chain-of-Thought Prompting

A faithful, minimal, self-contained reproduction of the central idea of
Wei et al., *"Chain-of-Thought Prompting Elicits Reasoning in Large Language
Models"* (2022), arXiv:2201.11903. Everything needed to read, run, and
understand the idea lives in **this folder**: the paper PDF, a from-scratch
implementation, a runnable CPU demo, and an interactive visualization.

## Plain-English summary

Large language models often fail at multi-step problems when asked to produce
the answer directly, but succeed when first prompted to write out the
intermediate reasoning steps — a *chain of thought*. Generating those steps
gives the model extra tokens of "working memory" and turns one hard global
computation into a sequence of easy local ones.

We cannot run a giant LLM on a CPU, so this demo reproduces the *mechanism* at
toy scale. We train **the same tiny decoder-only Transformer** two ways on
multi-digit **addition**:

- **Direct**: learn to emit the answer immediately — `12+345=357`.
- **Chain-of-thought**: learn to emit a digit-by-digit scratchpad with explicit
  carries, then the answer —
  `12+345=2+5+0=7c0|1+4+0=5c0|0+3+0=3c0#357`.

Direct addition is genuinely hard for a small model: carries propagate
right-to-left, but the answer is written left-to-right, so it must "look
ahead". The chain-of-thought decomposition makes every step *local*
(two digits + a carry → one digit + a carry), which the model learns easily.

**This is a simplification of the paper**: the paper elicits reasoning from a
frozen 100B+ model purely through the *prompt*, with no training. Here we
*train* a tiny model on each answer format instead, because that is what fits on
a CPU. The takeaway is identical — writing the reasoning out first produces
dramatically higher accuracy.

## What the demo shows

Trained for ~1,100 steps each (about 50 seconds total on CPU):

| Model | Overall exact-match | 1-digit | 2-digit | 3-digit |
|---|---|---|---|---|
| Direct | ~4% | ~0% | ~5% | ~4% |
| Chain-of-thought | **~93%** | ~100% | ~99% | ~89% |

The direct model fails at every size and collapses on longer problems; the
chain-of-thought model is near-perfect. The demo prints the accuracy table plus
a real generated scratchpad, and writes `data/cot_results.json` for the viz.

## Folder layout

```
Chain-of-Thought Prompting/
├── chain-of-thought_prompting.pdf   # the paper
├── requirements.txt                 # pinned deps (CPU PyTorch)
├── src/                             # from-scratch implementation
│   ├── model.py                     #   tiny decoder-only Transformer (TinyGPT)
│   ├── tokenizer.py                 #   character-level vocabulary
│   └── arithmetic.py                #   DIRECT vs CHAIN-OF-THOUGHT formatting
├── data/                            # sample data + generated results
│   └── generate_data.py             #   writes a tiny human-readable sample
├── demo/
│   └── run_demo.py                  #   trains both models & compares (CPU, <60s)
└── visualization/
    └── index.html                   # interactive, colorful, offline
```

## Setup

Requires Python 3.10+. Reuse the shared virtual environment created for the
reference paper (no new venv needed):

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
python data/generate_data.py     # (optional) write data/sample_dataset.json
python demo/run_demo.py          # train both models, compare, ~50s on CPU
```

Everything is deterministic (`--seed`). Flags let you change size/length:
`python demo/run_demo.py --help`.

## Expected output

```
================ EXACT-MATCH ACCURACY (held-out) ================
  DIRECT :   4.4%
  CoT    :  92.5%
  ...
  Worked example  602+26 = 628
    direct -> 643   (answer 643)
    cot    -> 2+6+0=8c0|0+2+0=2c0|6+0+0=6c0#628
             answer 628
OK: chain-of-thought scratchpad beats direct answering.
```

## Explore the visualization

Open `visualization/index.html` in any browser (works offline via `file://`).
It shows the two reasoning paths side by side, the accuracy comparison as bars
and a training curve, and a real generated scratchpad rendered step by step.
The page ships with baked-in results; served over HTTP it reloads live data:

```bash
python -m http.server 8000   # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper idea | Where it lives | File |
|---|---|---|
| Standard prompting (answer only) | `"direct"` target format | `src/arithmetic.py` |
| Chain-of-thought (reasoning then answer) | `"cot"` scratchpad format | `src/arithmetic.py` |
| Decomposing a task into intermediate steps | digit-by-digit carries | `src/arithmetic.py` → `cot_scratchpad` |
| The language model being prompted | tiny decoder-only Transformer | `src/model.py` → `TinyGPT` |
| Exact-match evaluation of reasoning | greedy decode + parse answer | `demo/run_demo.py` |
