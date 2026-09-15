# Chain of Draft

**Chain of Draft: Thinking Faster by Writing Less** — Silei Xu, Wenhao Xie, Lingxiao Zhao,
Pengcheng He (Zoom Communications), 2025. arXiv:2502.18600.

A faithful, minimal, fully self-contained reproduction of the paper's central
claim. Everything needed to read, run, and understand it lives in **this
folder**: the paper PDF, a from-scratch implementation, a runnable demo with
data, and an interactive visualization.

## Plain-English summary

Chain-of-Thought (CoT) prompting makes language models solve hard problems by
"thinking out loud" — writing every reasoning step in full prose. It works, but
those words cost tokens (money, latency). **Chain of Draft (CoD)** keeps the
*reasoning* but strips the *prose*: each step is written as a minimal draft (a
bare equation or a few symbols) instead of a sentence. The paper shows CoD
**matches CoT accuracy while using a small fraction of the tokens**.

The key insight this demo isolates: the accuracy gain of CoT comes from
*showing the intermediate steps at all*, not from the wordiness of those steps.
So a terse draft carries the same information for far less.

## What the demo shows

One small decoder-only Transformer (`TinyGPT`) is trained on a **4-step chained
addition mod 10** task rendered in three prompting styles, then evaluated on
held-out problems by prompting the *same* model each way:

| Style | Example completion for `2+9+8+8` | What it is |
|---|---|---|
| **Standard** (direct) | `#7.` | answer only, no reasoning |
| **Chain of Thought** | `2 plus 9 is 1, 1 plus 8 is 9, 9 plus 8 is 7, answer is 7#7.` | verbose prose per step |
| **Chain of Draft** | `2+9=1 1+8=9 9+8=7#7.` | one terse equation per step |

The demo reports **accuracy and mean tokens** for each style. Expected result:
direct answering is weak; CoT and CoD are both ~99–100% accurate; and **CoD uses
about a third of CoT's tokens** for the same answers.

Why direct answering is weak: composing several dependent `mod 10` steps in one
shot has nowhere to store the running sum, so a tiny model struggles. Emitting
intermediate results turns it into a chain of trivial `(a+b) mod 10` lookups.
This is exactly the CoT/CoD advantage, at small scale. (The unlearnable direct
target is down-weighted during training so it can't swamp the shared model —
see the note in `demo/run_demo.py`.)

## Folder layout

```
Chain of Draft/
├── chain_of_draft.pdf          # the paper itself
├── requirements.txt            # pinned deps (CPU PyTorch + numpy)
├── src/
│   ├── task.py                 # the task + 3 style renderers (Standard/CoT/CoD)
│   └── model.py                # TinyGPT: a small causal Transformer
├── data/
│   ├── generate_data.py        # writes sample examples.json (illustrative)
│   ├── examples.json           # same problems in all 3 styles (generated)
│   └── cod_run.json            # accuracy/token results for the viz (generated)
├── demo/
│   └── run_demo.py             # train one model on all 3 styles; eval + JSON
└── visualization/
    └── index.html              # colorful, interactive comparison
```

## Setup

Reuses the shared CPU virtual environment in the reference folder (no GPU, no
internet needed at runtime):

```bash
source "/workspace/Attention Is All You Need/.venv/bin/activate"
```

To create a fresh one instead (Python 3.10+):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## How to run

```bash
python demo/run_demo.py
```

Trains the model on all three styles and evaluates on held-out problems. Runs in
**~35 s on CPU** and writes `data/cod_run.json` for the visualization. Optionally
regenerate the illustrative sample set with `python data/generate_data.py`.

## Expected output

```
Held-out results (greedy decode):
  style               accuracy   mean tokens
  Standard (direct)    11.3%         3.0
  Chain of Thought    100.0%        59.0
  Chain of Draft       99.3%        20.0

  → CoD uses 34% of CoT's tokens (20.0 vs 59.0) at accuracy 99% vs 100%.
OK: Chain of Draft matches Chain of Thought accuracy with far fewer tokens.
```

The demo asserts CoT and CoD accuracy > 85%, CoD beats direct answering by a
wide margin, and CoD uses < half of CoT's tokens.

## Visualization

Open `visualization/index.html` in any browser (works offline via `file://`).
It shows:

- the **same problem answered three ways**, with each style's real completion,
  accuracy, and token count captured from the demo run;
- the **shared reasoning skeleton** and equations that make CoT and CoD identical
  in logic but different in verbosity;
- an animated **token race** — play the two chains step by step and watch CoD's
  running token count stay far below CoT's while reaching the same answer;
- **accuracy vs. token-cost bars** across all three styles, plus headline metrics.

Real run data is baked into the page; serve the folder
(`python -m http.server`) to load live `data/cod_run.json` instead.

## Code ↔ paper map

| Paper section | Component | File |
|---|---|---|
| §2, §3.1 | Three prompting styles (Standard / CoT / CoD) | `src/task.py` (`render_standard`, `render_cot`, `render_cod`) |
| §3.1 | Chain-of-Draft terse per-step rendering | `src/task.py` (`render_cod`) |
| — | One model answers all styles | `src/model.py` (`TinyGPT`) |
| §4 (accuracy & efficiency) | Train + eval accuracy and token counts | `demo/run_demo.py` |

## Honest scope notes

This is a *small-scale mechanism reproduction*, not the paper's LLM-scale
evaluation. The paper prompts large pretrained models (GPT-4o, Claude) on
benchmarks like GSM8K and shows CoD matches CoT accuracy at ~20% of the tokens.
Here we train a tiny model from scratch on a toy multi-step task to isolate the
same effect end-to-end on CPU: reasoning scaffolding drives accuracy, and the
terse draft form preserves it at a fraction of the token cost. "Tokens" here are
characters of this char-level model; the *ratio* between styles is the point.
