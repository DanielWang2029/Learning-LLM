# Search-R1 — RL for Interleaved Reasoning + Search

A faithful, minimal, self-contained reproduction of the signature idea of Jin et
al., *"Search-R1: Training LLMs to Reason and Leverage Search Engines with
Reinforcement Learning"* (2025), arXiv:2503.09516. Everything needed to read, run,
and understand the idea lives in **this folder**: the paper PDF, a from-scratch
implementation, a runnable CPU demo, and an interactive visualization.

## Plain-English summary

Search-R1 trains a model, with **reinforcement learning**, to interleave its own
reasoning with calls to a real **search engine**:

```
<search> query </search> -> <information> results </information> -> ... -> <answer> ... </answer>
```

The reward is just whether the final answer is correct — the model discovers on
its own *when* to search and *what* to search for. Because searching is a
*procedure* rather than a memorized fact, the learned behaviour **generalizes to
questions about entities it never trained on**.

## What this demo does

We build a small **corpus** of facts and a real **retrieval tool** over it. Answers
require **multi-hop** search:

```
Question: What country does Person_p live in?
  hop 1: search Person_p  ->  Person_p lives_in City_c  (+ born_in / likes distractors)
  hop 2: search City_c    ->  City_c located_in Country_k
  answer: Country_k
```

A tiny **policy** is trained with **REINFORCE + a GRPO-style group baseline** (no
value network) to issue searches and answer. Its action scores depend only on
**entity type** (person/city/country/…), never identity — so the learned search
procedure transfers to held-out persons. We compare it against a **no-search
classifier** that maps a person directly to a country: it can only memorize the
training persons and fails on the held-out set.

## What the demo shows

```
Search policy (before RL) test accuracy: 0.0%
  step    0 | train acc   0.0% | avg searches 0.00
  step   20 | train acc 100.0% | avg searches 5.00
  step   70 | train acc 100.0% | avg searches 3.00
  step   95 | train acc 100.0% | avg searches 2.00

                          train persons    held-out persons
  WITH search (RL)         100.0%           100.0%   (2.00 searches/q)
  NO search (classifier)   100.0%            10.0%

Interleaved trace on a HELD-OUT question:
Question: What country does Person_40 live in?
  <search>Person_40</search>
  <information>Person_40 lives_in City_1; Person_40 born_in Year_1980; Person_40 likes Color_4</information>
  <search>City_1</search>
  <information>City_1 located_in Country_4</information>
  <answer>Country_4</answer>
  correct answer: Country_4 -> OK
```

From reward alone the policy learns to (1) search before answering, (2) follow the
2-hop `person → city → country` chain, and (3) **skip the distractor facts**
(dropping from 5 searches to 2). On **held-out** persons, search scores **100%**
vs the memorizing baseline's **10%**. Runs in ~2s on CPU. Results are written to
`data/search_r1_results.json` for the viz.

## Folder layout

```
Search-R1/
├── search-r1.pdf             # the paper
├── requirements.txt          # pinned deps (CPU PyTorch)
├── src/
│   ├── corpus.py             # tiny fact KB + the retrieval "search" tool
│   ├── policy.py             # RL search policy + no-search baseline
│   └── grpo.py               # group-relative advantage (GRPO baseline)
├── data/
│   └── generate_data.py      # writes a readable slice of the corpus
├── demo/
│   └── run_demo.py           # RL training + search-vs-no-search comparison
└── visualization/
    └── index.html            # RL loop + interleaved search trace + generalization
```

## Setup

Requires Python 3.10+. Reuse the shared virtual environment:

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
python data/generate_data.py    # (optional) write data/sample_dataset.json
python demo/run_demo.py         # RL training + comparison, ~2s on CPU
```

Deterministic (`--seed`). Try `--n-train`, `--group-size`, `--lr`, `--steps`.

## Explore the visualization

Open `visualization/index.html` in any browser (offline via `file://`). It shows
the RL loop (group rollouts → group-relative advantages), the interleaved
`<search>/<information>/<answer>` trace on a held-out question, and the
search-vs-no-search generalization gap. Ships with baked-in data; served over HTTP
it reloads live results:

```bash
python -m http.server 8000   # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper idea | Where it lives | File |
|---|---|---|
| Real search tool over a corpus | fact lookup | `src/corpus.py` → `Corpus.search` |
| Interleaved reason→search→answer | rollout loop | `src/policy.py` → `SearchPolicy.rollout` |
| RL from a correctness reward | REINFORCE + group baseline | `demo/run_demo.py`, `src/grpo.py` |
| Group-relative advantage (no value net) | standardize within group | `src/grpo.py` → `group_advantages` |
| Search generalizes; memorization doesn't | held-out comparison | `src/policy.py` → `NoSearchClassifier`, `demo/run_demo.py` |
