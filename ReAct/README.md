# ReAct

A faithful, minimal, self-contained reproduction of the core idea of
Yao et al., *"ReAct: Synergizing Reasoning and Acting in Language Models"*
(2022), arXiv:2210.03629. Everything needed to read, run, and understand the
idea lives in **this folder**: the paper PDF, a from-scratch implementation, a
runnable CPU demo, and an interactive visualization.

## Plain-English summary

Chain-of-thought reasons but never *acts*; a pure tool-using agent acts but
can't *reason* about what it sees. **ReAct** interleaves the two: the agent
emits a **Thought**, takes an **Action** (a real tool call), reads the
**Observation**, and uses it to decide the next Thought/Action — looping until
it can **Finish** with an answer.

```
Thought → Action → Observation → Thought → Action → Observation → … → Finish
```

Because each action can depend on the previous observation, ReAct naturally
handles **multi-hop** questions: look up which country a person lives in, *then*
look up that country's capital.

## What this demo does (and how it simplifies the paper)

We build a tiny environment with **two real tools**:

- `Lookup[entity, attribute]` — query a small key-value knowledge base
  (`Alice.country = France`, `France.capital = Paris`, `car.wheels = 4`, …).
- `Calc[expression]` — evaluate arithmetic (`68+214 = 282`).

Three agents answer the same multi-hop questions:

1. **ReAct** — reasons about each observation and chains tool calls.
2. **Reasoning-only** — may reason but has *no tools*; it falls back on flawed
   parametric memory and **hallucinates**.
3. **Acting-only** — may call tools but does *no reasoning to chain them*; it
   fires a single naive action and fails on multi-hop questions.

**Honesty note:** the policy is a transparent **rule-based** controller, not an
LLM — but the tools and observations are genuinely computed, and the ReAct agent
really does feed each observation into its next action. The point the paper
makes is reproduced exactly: reasoning **and** acting together beat either
alone.

## What the demo shows

```
==================== ACCURACY ====================
  ReAct           : 6/6  (100%)
  Reasoning-only  : 1/6  (17%)
  Acting-only     : 1/6  (17%)

=== Why tools + reasoning are BOTH needed ===
Q: What is the capital of the country Bob lives in?  (gold Tokyo)
  ReAct            -> Tokyo        ✓
  Reasoning-only   -> Paris        ✗   (hallucinated from memory)
  Acting-only      -> unknown      ✗   (one naive lookup, no chaining)
```

It also prints the **full interleaved Thought/Action/Observation trace** for
every question and writes `data/react_results.json` for the visualization.

## Folder layout

```
ReAct/
├── react.pdf                    # the paper
├── requirements.txt             # (demo is pure stdlib; numpy optional)
├── src/
│   ├── environment.py           # knowledge base + the two real tools
│   ├── questions.py             # the multi-hop question set
│   └── agents.py                # ReAct, reasoning-only, acting-only
├── data/
│   └── generate_data.py         # serializes the KB + questions
├── demo/
│   └── run_demo.py              # runs all three agents, prints traces
└── visualization/
    └── index.html               # animated Thought/Action/Observation trace
```

## Setup

Requires Python 3.10+. The demo needs only the standard library; you can also
reuse the shared environment:

```bash
source "../Attention Is All You Need/.venv/bin/activate"
```

## How to run

```bash
python data/generate_data.py    # (optional) write data/environment.json
python demo/run_demo.py         # traces + accuracy comparison
```

## Explore the visualization

Open `visualization/index.html` in any browser (offline via `file://`). It
animates the real Thought → Action → Observation loop for each question, shows
the three-way ablation on a multi-hop question, and the accuracy bars. Ships with
baked-in data; served over HTTP it reloads live results:

```bash
python -m http.server 8000   # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper component | Where it lives | File |
|---|---|---|
| Acting in an environment (tools) | KB lookup + calculator | `src/environment.py` |
| Thought / Action / Observation loop | interleaved trace | `src/agents.py` → `react_agent` |
| Reasoning-only ablation (CoT) | no-tool agent | `src/agents.py` → `reasoning_only` |
| Acting-only ablation (Act) | no-reasoning agent | `src/agents.py` → `acting_only` |
| Multi-hop QA task | question set | `src/questions.py` |
