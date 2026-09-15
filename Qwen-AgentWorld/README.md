# Qwen-AgentWorld — Language World Model (LWM)

A minimal, self-contained, CPU-only reproduction of Qwen-AgentWorld's core idea:
a **Language World Model** that predicts the next environment state from
`(state, action)`, so an agent can **plan and roll out trajectories in
imagination** instead of in the real environment.

- **Paper:** *Qwen-AgentWorld: Language World Models for General Agents*
- **Authors:** Alibaba (Qwen Team)
- **Year:** 2026 · **arXiv:** 2606.24597

> **Scope & honesty.** Qwen-AgentWorld is a large MoE world model trained on 10M+
> real trajectories across 7 agentic domains (MCP, Search, Terminal, SWE,
> Android, Web, OS) and evaluated with long chain-of-thought reasoning. That
> cannot be reproduced here. This demo reproduces the paper's *documented core
> mechanism* — learning `(state, action) → next state` and using it as a
> decoupled simulator for planning (Section 1–2) — from scratch on a tiny toy
> environment, so the mechanism runs end-to-end on CPU in seconds.

## What the demo shows

The paper's key distinction: a *policy* maps `state → action`, while a *world
model* maps `(state, action) → next state`. The demo:

1. **learns the dynamics** of a toy gridworld (with a wall) by predicting the
   next state from `(state, action)`, describing each state by its `(row, col)`
   tokens so the model generalizes the movement rule — measured on **held-out**
   transitions it never trained on;
2. **plans in imagination** — breadth-first search that expands states purely by
   querying the world model, never the real environment; and
3. **verifies in reality** — executes the imagined plan in the true environment
   and checks the imagined trajectory matches, step for step, and reaches the
   goal.

Representative output (seed 0):

```
next-state accuracy (all)      : 100.0%
next-state accuracy (held-out) : 100.0%
imagined plan reaches goal      : True
planning success over 58 pairs : 100.0%
```

The main plan detours the agent down and around the wall, then up to the goal —
and the imagined state sequence is identical to the real one.

## Folder layout

```
Qwen-AgentWorld/
├── qwen-agentworld.pdf       # the paper
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── env.py                # GridWorld — the "real" environment
│   └── world_model.py        # WorldModel (state,action → next state) + imagination planner
├── data/
│   ├── generate_data.py      # defines the grid + dumps the transition table
│   ├── env.json              # (generated)
│   └── world_model_result.json  # (generated) accuracy + plan + rollouts for the viz
├── demo/
│   └── run_demo.py           # trains, plans in imagination, verifies in reality
└── visualization/
    └── index.html            # world-model loop + animated planned path + rollout match
```

## Setup

Requires Python 3.10+. From inside this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## How to run

```bash
python demo/run_demo.py
```

Trains and plans in ~2s on CPU and writes `data/world_model_result.json` for the
visualization.

## Expected output

The demo asserts held-out next-state accuracy > 0.85 (reaches ~100%), that the
imagined plan reaches the goal and matches the real rollout exactly, and that
planning succeeds on > 90% of random start/goal pairs, then prints
`OK: learned world model predicts next states and plans in imagination.`

## Explore the visualization

Open `visualization/index.html` in any browser (offline via `file://`): the
world-model loop, an **animated agent** following the imagined plan around the
wall from start to goal (press play), and a side-by-side imagined-vs-real
rollout table showing they match. For live data, serve the folder:

```bash
python -m http.server 8000   # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper idea | Where | File |
|---|---|---|
| World model `(state, action) → next state` | prediction heads | `src/world_model.py` → `WorldModel` |
| State described by tokens (row, col) | input encoding | `src/world_model.py` → `forward` |
| Decoupled simulator for planning/RL | imagination BFS | `src/world_model.py` → `plan` |
| Simulation fidelity vs. real env | imagined-vs-real rollout | `demo/run_demo.py` |
| The environment being modeled | toy gridworld | `src/env.py` → `GridWorld` |
