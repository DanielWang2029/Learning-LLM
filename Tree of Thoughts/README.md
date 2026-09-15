# Tree of Thoughts

A faithful, minimal, self-contained reproduction of the core idea of
Yao et al., *"Tree of Thoughts: Deliberate Problem Solving with Large Language
Models"* (2023), arXiv:2305.10601. Everything needed to read, run, and
understand the idea lives in **this folder**: the paper PDF, a from-scratch
implementation, a runnable CPU demo, and an interactive visualization.

## Plain-English summary

Chain-of-thought produces a single, left-to-right line of reasoning. For
problems that need exploration and backtracking, that is fragile: one bad early
step dooms the whole chain. **Tree of Thoughts (ToT)** reframes problem solving
as **search over a tree** of partial solutions ("thoughts"), with three parts:

1. a **thought generator** that proposes candidate next steps,
2. a **state evaluator** that judges how promising each partial state is
   (*sure / maybe / impossible*), and
3. a deliberate **search** (BFS/DFS with a beam and pruning) that explores
   promising branches and abandons dead ones.

## The task: Game of 24

Reach **24** from four numbers using `+ − × ÷` (each number once). This is the
paper's headline example because a single chain of thought almost always fails,
while search succeeds. A state is the set of numbers still on the table; each
thought combines two of them into one:

```
(4 numbers) --op--> (3 numbers) --op--> (2 numbers) --op--> (1 number == 24?)
```

## What this demo does (and how it simplifies the paper)

The paper uses a large language model as both the generator and the evaluator.
On CPU we replace those with **transparent programmatic stand-ins**:

- **Generator** (`src/game24.py` → `moves`): enumerates every valid combination
  of two numbers (integer results only, exact division).
- **Evaluator** (`src/game24.py` → `evaluate`): a *sure / maybe / impossible*
  judgement. It is exact (and therefore a sound prune) for 1–2 numbers, but a
  deliberately imperfect shallow lookahead for 3+ numbers — which is exactly why
  a greedy chain gets misled.
- **Search** (`src/search.py` → `search`): BFS keeping the top-`beam` states per
  level and pruning impossible ones. The **greedy chain-of-thought baseline is
  the same search with beam width 1** (commit to the single best step, no
  backtracking) — the failure mode the paper highlights.

**Honesty note:** the generator/evaluator are rules, not an LLM, and division is
restricted to exact integers (instances are filtered to be solvable that way).
The mechanism — *deliberate search beats a single greedy chain* — is the point,
and it is reproduced exactly.

## What the demo shows

```
==================== SUCCESS RATES ====================
  greedy chain of thought (beam 1) :  31.0%  (31/100)
  tree of thoughts (beam 5)        :  77.0%  (77/100)
  avg nodes expanded by ToT        : 8.4

  Showcase puzzle: [3, 4, 4, 5]  ->  reach 24
    greedy (beam 1): FAILED (committed to a dead end)
    tree-of-thoughts: SOLVED  4x4=16 , 3+5=8 , 8+16=24
    ToT explored 97 nodes (83 pruned).
```

Tree of Thoughts solves ~77% of solvable puzzles vs ~31% for a greedy chain —
mirroring the paper's ToT ≈ 74% vs CoT ≈ 4%. Runs in a fraction of a second.

## Folder layout

```
Tree of Thoughts/
├── tree_of_thoughts.pdf         # the paper
├── requirements.txt             # (demo is pure stdlib; numpy optional)
├── src/
│   ├── game24.py                # generator (moves) + evaluator + exact solver
│   └── search.py                # BFS beam search; beam=1 is the greedy baseline
├── data/
│   └── generate_data.py         # writes a tiny sample of instances
├── demo/
│   └── run_demo.py              # success rates + a recorded search tree
└── visualization/
    └── index.html               # interactive/animated search tree, offline
```

## Setup

Requires Python 3.10+. The demo needs only the standard library; you can also
reuse the shared environment:

```bash
source "../Attention Is All You Need/.venv/bin/activate"
```

## How to run

```bash
python data/generate_data.py    # (optional) write data/sample_instances.json
python demo/run_demo.py         # success rates + record a search tree
```

Deterministic (`--seed`). Try `--beam` and `--n-instances`.

## Explore the visualization

Open `visualization/index.html` in any browser (offline via `file://`). It
animates the real recorded search tree — solution path, explored (beam) nodes,
and pruned branches — with a toggle between the greedy chain (beam 1) that fails
and the full tree-of-thoughts search that solves it, plus the success-rate bars.
Ships with baked-in data; served over HTTP it reloads live results:

```bash
python -m http.server 8000   # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper component | Where it lives | File |
|---|---|---|
| Thought generator (propose next steps) | enumerate number/operator combos | `src/game24.py` → `moves` |
| State evaluator (sure/maybe/impossible) | heuristic judgement + value | `src/game24.py` → `evaluate` |
| Deliberate search (BFS + beam + prune) | level-wise beam search | `src/search.py` → `search` |
| Chain-of-thought baseline | same search, beam width 1 | `src/search.py` (beam=1) |
| Recorded search tree for analysis | nodes + statuses + path | `demo/run_demo.py` |
