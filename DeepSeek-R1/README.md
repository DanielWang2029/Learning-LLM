# DeepSeek-R1 (GRPO)

A faithful, minimal, self-contained reproduction of the core idea of
DeepSeek-AI, *"DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via
Reinforcement Learning"* (2025), arXiv:2501.12948. Everything needed to read,
run, and understand the idea lives in **this folder**: the paper PDF, a
from-scratch implementation, a runnable CPU demo, and an interactive
visualization.

## Plain-English summary

DeepSeek-R1 shows that reasoning behaviour can be **incentivized by reinforcement
learning**, not just supervised imitation. A model is asked to answer inside a
`<think>…</think><answer>…</answer>` format and is rewarded only for (a) using
that format and (b) getting the answer right — there are **no supervised
reasoning traces**. Optimization uses **GRPO (Group-Relative Policy
Optimization)**: for each prompt, sample a *group* of outputs, use the group's
own mean reward as the baseline (so there is **no value network**), and nudge
above-average outputs up and below-average ones down.

```
sample group of G outputs per prompt
advantage_i = (reward_i - mean(rewards)) / (std(rewards) + ε)
loss = - mean_i [ advantage_i · log π(output_i | prompt) ]  - β · entropy
```

## What this demo does (and how it simplifies the paper)

We can't RL-train a real LLM on CPU, so we train a **small parameterized policy**
from scratch on a toy task: given `a+b=`, emit `{a+b}[answer]` — where `{...}`
stands in for `<think>…</think>` and `[...]` for `<answer>…</answer>`. The policy
makes three learnable decisions per response:

- **use tags** (global) — adopt the reasoning/answer format, or blurt a bare
  number;
- **show work** (global) — put the computation `a+b` inside the think block;
- **answer digit** (one distribution per prompt) — where the arithmetic is
  actually learned.

It is trained with real GRPO: group sampling, group-relative (standardized)
advantage, a REINFORCE policy gradient, an entropy bonus for exploration, and
**no critic**.

**Honesty note:** the "policy" is a handful of categorical distributions, not a
deep transformer, and the task is tiny. This keeps RL-from-scratch stable and
fast on CPU while faithfully exercising the GRPO update and reward design. The
result reproduces the paper's message: **from reward alone, the model learns to
use the reasoning format and to be correct.**

## What the demo shows

```
  step    0/400 (chance) | acc   0.0% | format   0.0% | think   0.0%
  step    1/400          | acc  32.0% | format 100.0% | think 100.0%
  step   15/400          | acc  76.0% | format 100.0% | think 100.0%
  step   30/400          | acc  92.0% | format 100.0% | think 100.0%
  step   45/400          | acc 100.0% | format 100.0% | think 100.0%
  ...
Final: accuracy 100.0% | format-usage 100.0% | think-usage 100.0% | mean reward 1.700

  0+0={0+0}[0]   1+2={1+2}[3]   3+4={3+4}[7]   4+4={4+4}[8]   (all correct)
```

Accuracy and mean reward climb from chance to the maximum, and the
`<think>`/`<answer>` format emerges — all from the scalar reward. Runs in a few
seconds. Results are written to `data/r1_results.json` for the viz.

## Folder layout

```
DeepSeek-R1/
├── deepseek-r1.pdf              # the paper
├── requirements.txt             # pinned deps (CPU PyTorch)
├── src/
│   ├── policy.py                # small parameterized policy (the RL-trained model)
│   ├── task.py                  # the {think}[answer] task + reward function
│   └── grpo.py                  # group-relative advantage + policy-gradient loss
├── data/
│   └── generate_data.py         # serializes the task + reward scheme
├── demo/
│   └── run_demo.py              # GRPO training loop (CPU, a few seconds)
└── visualization/
    └── index.html               # GRPO loop, reward curve, format emergence
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
python data/generate_data.py    # (optional) write data/task.json
python demo/run_demo.py         # GRPO training, ~3s on CPU
```

Deterministic (`--seed`). Try `--group-size`, `--lr`, `--entropy-coef`, `--steps`.

## Explore the visualization

Open `visualization/index.html` in any browser (offline via `file://`). It shows
the GRPO loop (group of rollouts → group-relative advantages), the reward /
accuracy / format-usage learning curves, and the before-vs-after emergence of the
reasoning format. Ships with baked-in data; served over HTTP it reloads live
results:

```bash
python -m http.server 8000   # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper component | Where it lives | File |
|---|---|---|
| `<think>…</think><answer>…</answer>` format | `{think}[answer]` response | `src/policy.py` → `compose` |
| Format reward + accuracy reward | reward function | `src/task.py` → `reward` |
| Group sampling | G outputs per prompt | `demo/run_demo.py` |
| Group-relative advantage (no value net) | standardize within group | `src/grpo.py` → `group_advantages` |
| Policy-gradient update | REINFORCE + entropy | `src/grpo.py` → `grpo_loss` |
| Emergent reasoning behaviour | rising accuracy/format | `demo/run_demo.py`, viz |
