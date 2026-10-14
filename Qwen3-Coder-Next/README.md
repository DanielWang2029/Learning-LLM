# Qwen3-Coder-Next

**Qwen Team · 2026 · arXiv:2603.00729**

Qwen3-Coder-Next is a coding-agent model based on Qwen3-Next's hybrid
attention and Mixture-of-Experts (MoE) architecture.  Its report emphasizes
large-scale agentic training on verifiable coding tasks: the agent proposes a
solution, a reproducible environment executes tests, and reinforcement learning
uses that execution feedback.  The full model contains 80B parameters while
activating 3B per inference pass.

This folder is a tiny CPU-only reproduction of the **mechanism**, not of that
model, its hybrid-attention implementation, task scale, environments, or
benchmark results.  A small task-conditioned policy with a genuine top-2-of-4
MoE feed-forward block emits one-line programs.  A deterministic executor runs
hidden unit tests; group-relative test rewards drive a compact GRPO-style
policy-gradient update.

## What the demo shows

The demo reports greedy unit-test pass rate before and after training, a
pass-rate curve, and MoE total versus active parameter counts.  It also prints
a concrete emitted program and its unit-test outputs.  The verifier is the
sole reward source: no target-program label appears in the reward calculation.

## Folder layout

```text
Qwen3-Coder-Next/
├── qwen3-coder-next.pdf       # supplied paper
├── requirements.txt           # CPU-only PyTorch and NumPy
├── src/agent.py               # sparse top-k MoE policy and executable verifier
├── data/generate_data.py      # specs and hidden unit-test inputs
├── demo/run_demo.py           # GRPO-style rollouts, updates, and metrics
└── visualization/index.html   # offline interactive agentic-RL explanation
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## Run

```bash
python demo/run_demo.py
```

On CPU the deterministic seeded run completes in a few seconds.  Expected
output rises from `5.6%` to `100.0%` greedy hidden-test pass rate,
reports `16,896` total MoE parameters versus `8,512` active per decision, and
writes the measured curve and example to `data/agentic_rl_results.json`.

Open `visualization/index.html` directly in a browser for the offline,
data-baked view.  If served with `python -m http.server`, its Load button can
read the freshly written JSON file.

## Code ↔ paper map

| Paper section | Reproduced idea | File |
|---|---|---|
| §1 | Sparse MoE with a small active parameter footprint | `src/agent.py` |
| §2 | Verifiable executable tasks and execution feedback | `data/generate_data.py`, `src/agent.py` |
| §2.1 | Program proposal verified by hidden tests | `src/agent.py` |
| §4.2.3 | Execution-driven RL using unit-test rewards | `demo/run_demo.py` |
| §5 | Pass rate as a coding-agent outcome metric | `demo/run_demo.py` |
