# GPT-3

A faithful, minimal, CPU-friendly reproduction of the key idea from **GPT-3** —
*"Language Models are Few-Shot Learners"* (Brown et al., 2020;
[arXiv:2005.14165](https://arxiv.org/abs/2005.14165)). Everything needed to read,
run and understand **in-context few-shot learning** lives in this folder.

## The idea in two paragraphs

GPT-3 is again the same decoder-only Transformer, scaled up enormously (175B
parameters). Its headline finding is **in-context learning**: you can teach the
model a new task at *inference time*, with **no weight updates**, simply by
putting a few input→output demonstrations in the prompt followed by a query. The
model infers the pattern from the demonstrations and completes the query.

Crucially, performance improves as you add more demonstrations ("shots"):
0-shot < 1-shot < few-shot. This repo reproduces exactly that trend at tiny
scale. We train a small LM on prompts drawn from a *family* of rules (overlapping
permutations of a symbol alphabet). Because the rules overlap, a single
demonstration is usually ambiguous, so the model must accumulate evidence across
several shots to identify the rule — and then apply it to a query symbol it was
never shown.

## What the demo shows

`demo/run_demo.py` trains the LM on demonstration prompts, then (weights frozen)
measures accuracy as a function of the number of shots K. It prints an
**accuracy-vs-shots curve** rising from near the marginal baseline at K=0 to
~90%+ at K=6 — a **+45-point** gain from context alone — and shows a worked
few-shot example the model solves correctly.

## Folder layout

```
GPT-3/
├── gpt-3.pdf               # the paper
├── requirements.txt        # torch (CPU) + numpy
├── src/                    # the implementation
│   ├── attention.py        #   causal self-attention (induction-head behaviour)
│   └── model.py            #   decoder-only LM
├── data/
│   └── generate_data.py    #   the rule family + prompt formatting
├── demo/
│   └── run_demo.py         #   train, then measure accuracy vs shots
└── visualization/
    └── index.html          # interactive, offline curve + rule table + example
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## Run

```bash
python data/generate_data.py     # optional: prints the rule family
python demo/run_demo.py          # ~1 minute on CPU
```

### Expected output

```
== In-context few-shot accuracy (no weight updates) ==
  K=0  acc  49.0%  |####
  K=1  acc  64.0%  |#####
  K=2  acc  80.0%  |######
  ...
  K=6  acc  93.7%  |#######
0-shot 49.0%  ->  6-shot 93.7%  (+44.7 pts from context alone)
OK: accuracy rises with the number of in-context demonstrations.
```

It writes `data/gpt3_sample.json` (the accuracy-vs-shots curve, the rule family,
a worked example) for the visualization.

## Explore the visualization

Open `visualization/index.html` in any browser (offline via `file://`). See the
accuracy-vs-shots curve with a chance line, the overlapping rule family table,
and a worked K-shot prompt with demonstrations, query and completion colour-coded.
For live data: `python -m http.server 8000` → `http://localhost:8000/visualization/`.

## Code ↔ paper map

| Paper concept | Component | File |
|---|---|---|
| Decoder-only LM | `GPT` | `src/model.py` |
| Induction / in-context mechanism | causal self-attention | `src/attention.py` |
| Few-shot demonstrations in the prompt (§2.1) | `make_sequence`, `build_prompt` | `data/generate_data.py`, `demo/run_demo.py` |
| Accuracy rising with K shots (Fig. 1.2) | `accuracy_at_k` sweep | `demo/run_demo.py` |
| No gradient updates at inference | frozen-weights evaluation | `demo/run_demo.py` |
