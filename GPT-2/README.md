# GPT-2

A faithful, minimal, CPU-friendly reproduction of **GPT-2** — *"Language Models
are Unsupervised Multitask Learners"* (Radford, Wu, Child, Luan, Amodei &
Sutskever, 2019). Everything needed to read, run and understand the core idea
lives in this folder.

## The idea in two paragraphs

GPT-2 is architecturally almost identical to GPT-1 (a decoder-only Transformer
with pre-LayerNorm blocks), just much larger. Its contribution is a
**behavioural** observation: a language model trained only to predict the next
token — with no task-specific supervision or output heads — can perform many
tasks **zero-shot**, provided the task is expressed *in the text itself*.

If the training text contains lines like `reverse: cafe = efac`, then at test
time you can prompt the model with `reverse: <new word> =` and the most likely
continuation *is* the answer. The task is just a conditional distribution the
LM already models. GPT-2 showed this ability grows sharply with scale; this repo
demonstrates the mechanism at tiny scale.

## What the demo shows

`demo/run_demo.py` trains a small character-level LM on a corpus of lines
`task: input = output;` for three string-transform tasks (**reverse**,
**upper**, **sort**), using only the next-token objective. Then, with weights
frozen, it evaluates the model **zero-shot** on freshly sampled, unseen inputs
by prompting `task: input =` and greedily decoding. It reports per-task accuracy
(typically ~90–97%) — one model, no task-specific heads.

## Folder layout

```
GPT-2/
├── gpt-2.pdf               # the paper
├── requirements.txt        # torch (CPU) + numpy
├── src/                    # the implementation
│   ├── attention.py        #   causal (masked) multi-head self-attention
│   └── model.py            #   GPT-2 decoder LM (pre-LN) + generate()
├── data/
│   └── generate_data.py    #   writes the multitask char corpus
├── demo/
│   └── run_demo.py         #   train LM, evaluate zero-shot
└── visualization/
    └── index.html          # interactive, offline task cards + accuracy + attention
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## Run

```bash
python data/generate_data.py     # optional: writes data/corpus.json to inspect
python demo/run_demo.py          # ~1 minute on CPU
```

### Expected output

```
== Zero-shot evaluation on unseen inputs (no fine-tuning) ==
   reverse: acc  96.7%   e.g. 'ebg' -> 'gbe' (want 'gbe')
     upper: acc  95.0%   e.g. 'fchcd' -> 'FCHCD' (want 'FCHCD')
      sort: acc  90.0%   e.g. 'afa' -> 'aaf' (want 'aaf')
Mean zero-shot accuracy across tasks: 93.9%
OK: a single next-token LM performs every task zero-shot.
```

It writes `data/gpt2_sample.json` (per-task accuracy, examples, an attention
matrix) for the visualization.

## Explore the visualization

Open `visualization/index.html` in any browser (offline via `file://`). See the
three tasks framed as text, per-task zero-shot accuracy bars, the LM loss curve,
and a causal self-attention heatmap. For live data: `python -m http.server 8000`
→ `http://localhost:8000/visualization/`.

## Code ↔ paper map

| Paper concept | Component | File |
|---|---|---|
| Decoder-only LM, pre-LN (§2) | `GPT`, `Block` | `src/model.py` |
| Causal self-attention | `CausalSelfAttention` | `src/attention.py` |
| Tasks framed as text (§3.1) | `task: input = output;` corpus | `data/generate_data.py` |
| Zero-shot evaluation (no heads) (§3) | prompt `task: input =`, decode | `demo/run_demo.py` |
| Unsupervised multitask learning | one LM, many tasks | `demo/run_demo.py` |
