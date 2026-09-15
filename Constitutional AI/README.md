# Constitutional AI — Harmlessness from AI Feedback

A faithful, minimal, fully self-contained reproduction of Bai et al.,
*"Constitutional AI: Harmlessness from AI Feedback"* (2022),
arXiv:[2212.08073](https://arxiv.org/abs/2212.08073).

Constitutional AI aligns a model to an explicit **constitution** (a set of
written principles) **without human preference labels**. It has two stages:

1. **SL-CAI (self-critique + revision):** the model looks at its own response,
   **critiques** it against the constitution, and **revises** it to comply —
   iterating until it does.
2. **RLAIF:** the revised responses are labeled *preferred* over the originals
   (this is the "AI feedback"), and a **preference model** is trained on those
   comparisons — the same reward model an RLHF pipeline would use, but with AI
   labels instead of human ones.

To keep everything reproducible on CPU, we run this on **toy text**: a tiny
vocabulary of "safe" and "toxic" words, and a three-rule constitution. The
critique/revision steps are deterministic rule functions (standing in for an
LLM asked to self-critique), so no large model or human is needed.

```
Constitutional AI/
├── constitutional_ai.pdf        # the paper itself
├── requirements.txt             # pinned CPU dependencies (torch, numpy)
├── src/                         # the method, from scratch
│   ├── words.py                 #   toy vocabulary + tokenizer
│   ├── constitution.py          #   the rules, critique, and revision loop (SL-CAI)
│   ├── generator.py             #   a toy model that sometimes violates the rules
│   └── preference_model.py      #   RLAIF preference model (tiny Transformer) + BT loss
├── data/                        # results JSON written by the demo (generated)
├── demo/run_demo.py             # critique/revise + RLAIF end-to-end, with evidence
└── visualization/index.html    # constitution, critique→revise trace, RLAIF curves
```

## 1. Set up the environment

Requires Python 3.10+. Reuse the shared virtual environment:

```bash
source "/workspace/Attention Is All You Need/.venv/bin/activate"
```

Or build a fresh one:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## 2. Run the demo

```bash
python demo/run_demo.py
```

It prints the constitution, then:

- **SL-CAI** — samples 1000 responses, runs the critique→revise loop, and
  reports compliance and per-principle violation counts **before vs after**.
- **RLAIF** — builds (revised ≻ original) preference pairs and trains the
  preference model with the Bradley-Terry loss, reporting how well it ranks
  compliant responses above non-compliant ones.

## 3. Expected output

Seeded and reproducible. A representative run (≈6 s on CPU):

```
compliance BEFORE critique/revise:  17.2%
compliance AFTER  critique/revise: 100.0%
per-rule violations (before -> after):
  [harmless      ]  420 ->    0
  [non-repetitive]  531 ->    0
  [concise       ]  325 ->    0
...
preference model — AI-pair ranking accuracy    : 100.0%
preference model — compliance-prediction acc.  :  92.5%

OK: critique+revise removed violations and RLAIF preference model works.
```

The proof it works:

1. **critique + revision eliminates every constitutional violation** (compliance
   goes from ~17% to 100%), and
2. the **preference model trained purely on AI feedback predicts compliance**
   with high accuracy — including the order- and length-based rules that a
   bag-of-words model cannot see (hence the tiny Transformer). The demo
   `assert`s all of this.

## 4. Explore the visualization

Open `visualization/index.html` in any browser (offline via `file://`). It
renders the constitution, a real step-by-step **critique→revise trace**,
before/after compliance (overall and per principle), and the RLAIF preference
model's learning curves — all baked in from your demo run. Serve the folder to
load live data:

```bash
python -m http.server 8000   # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper concept | Component | File |
|---|---|---|
| Constitution (principles) | Rule definitions | `src/constitution.py` → `CONSTITUTION` |
| Critique step | Find first violated principle | `src/constitution.py` → `first_violation` |
| Revision step | Rewrite to comply, iterate | `src/constitution.py` → `critique_and_revise` |
| Helpful-only model | Toy response generator | `src/generator.py` |
| RLAIF preference model | Transformer reward model | `src/preference_model.py` → `PreferenceModel` |
| Preference training | Bradley-Terry on AI labels | `src/preference_model.py` → `preference_loss` |
