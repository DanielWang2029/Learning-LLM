# Kimi k1.5 — RL for Reasoning with a Length Penalty

A minimal, CPU-only reproduction of one of the signature techniques in
**"Kimi k1.5: Scaling Reinforcement Learning with LLMs"**, Kimi Team — 2025,
[arXiv:2501.12599](https://arxiv.org/abs/2501.12599): reinforcement-learning for
reasoning with a **length penalty** that curbs overlong chains ("overthinking").

## Plain-English summary

Kimi k1.5 scales RL to train strong reasoning models. A recurring problem in
reasoning RL is that, optimized for correctness alone, models learn to produce
**longer and longer** chains of thought — more tokens, more cost, but not
proportionally more accuracy. Kimi's fix is a **length penalty** added to the RL
reward (paper §2.5). Within each sampled *group* of responses to a problem, it
rewards **shorter correct** answers and never rewards length on incorrect ones:

```
lambda_i     = 0.5 − (len_i − min_len) / (max_len − min_len)     # shorter ⇒ higher
len_reward_i = lambda_i            if the answer is correct
             = min(0, lambda_i)    otherwise
reward_i     = correctness_i + w · len_reward_i                  # w = 0 disables it
```

Combined with a **GRPO-style** update (group-relative advantage, no value
network), this keeps chains short while preserving — and even improving —
accuracy. Kimi calls the broader agenda of turning long chains into short ones
**"long2short"**.

## What the demo shows

We give a tiny policy a toy reasoning task — output `max(a, b)` — where it makes
two decisions per problem:

- an **answer** (the reasoning output, which must become correct), and
- a **chain length**: how many scratch/"think" tokens `T` to emit before the
  answer (the "thinking budget").

Crucially, thinking is *useful up to a point*: the answer is only reliably
**revealed** after `THINK_NEED` (=3) think steps; below that the model is forced
to guess, but thinking *beyond* 3 adds nothing. So the ideal is a **small
positive** chain length — not zero, and not the maximum. This is what makes the
length penalty a *control* knob rather than an off-switch.

The demo runs the Kimi pipeline: an **SFT-style warm-start** trains the reasoner
at full reveal (chain held long), then two RL copies from that **same** warmed
policy tune only the thinking budget and are compared:

- **no length penalty** (`w = 0`): reward = correctness only.
- **length penalty** (`w > 0`): reward = correctness + Kimi's group length reward.

Reference run (800 warm-start + 400 GRPO iterations each, ~8s on CPU):

```
                       accuracy    mean chain length
  before training        7.0%          (chance)
  after warm-start      76.0%          7.42     <- reasoner accurate, but overthinking
  RL no-penalty         76.0%          7.92     <- chains stay long
  RL length-penalty     76.0%          3.00     <- trimmed to the ~3 steps needed, same accuracy
```

The warm-start makes the reasoner accurate but overthinking. Without the penalty,
RL leaves the chain long — RL alone does **not** curb overthinking. With the
penalty the chain is trimmed to the minimum actually needed (≈`THINK_NEED`) at
**identical** accuracy: the "long2short" effect.

## Folder layout

```
Kimi K1.5/
├── kimi_k15.pdf               # the paper
├── requirements.txt           # pinned deps (CPU PyTorch + numpy)
├── src/
│   ├── model.py               #   two-headed policy (answer head + length policy)
│   ├── task.py                #   the toy task + completion rendering
│   └── rl.py                  #   §2.5 GRPO-style update + Kimi length-penalty reward
├── data/
│   └── generate_data.py       #   materializes the problem set to JSON
├── demo/
│   └── run_demo.py            #   warm-start + two RL runs (penalty vs none), curves
└── visualization/
    └── index.html             #   RL loop + accuracy/length learning curves
```

## Setup

Requires Python 3.10+. Reuse the shared virtual environment, or create one:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## How to run

```bash
python data/generate_data.py    # (optional) writes data/problems.json
python demo/run_demo.py          # warm-start + two RL runs + curves (~8s on CPU)
```

The demo writes `data/kimi_run.json` (learning curves + final metrics) for the
visualization.

## Expected output

```
before training: accuracy 7.0% (chance)
after warm-start: accuracy 76.0% | mean chain length 7.42
RL no-penalty     76.0%   len 7.92
RL length-penalty 76.0%   len 3.00
OK: length penalty controls chain length while accuracy stays high; RL alone leaves chains long.
```

## Explore the visualization

Open `visualization/index.html` in any browser (works offline via `file://`).
It diagrams the GRPO RL loop with the length-penalty reward, then plots the real
**accuracy** and **mean-chain-length** learning curves for both runs — showing
accuracy climbing in both while only the penalized run keeps length under
control. To load fresh data, serve the folder:

```bash
python -m http.server 8000    # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper idea | Where | File |
|---|---|---|
| RL for reasoning (policy improved by reward) | the RL loop | `demo/run_demo.py`, `src/rl.py` |
| GRPO-style group-relative advantage | `(r − mean)/std` within a group | `src/rl.py` → `grpo_step` |
| Length penalty (§2.5) | Kimi's group length reward | `src/rl.py` → `compute_rewards` |
| long2short (short chains, same accuracy) | penalty vs no-penalty comparison | `demo/run_demo.py` |

## Honest scope notes

This is a *mechanism* reproduction, not the Kimi system. The real Kimi k1.5 is a
large multimodal model trained with long-context RL, partial rollouts, a policy
optimization with a reference-model term, curriculum and sampling strategies, and
much more; the length penalty is one component. Here we isolate that component on
a synthetic task.

To study the length penalty cleanly on CPU, the policy exposes the **chain
length as an explicit action** (a length head) alongside the answer head, rather
than emitting an autoregressive stream where a single head both thinks and
answers. Two deliberate design choices make the demonstration faithful:

- **Reasoning is useful up to a threshold.** The answer is fully revealed only
  once the chain reaches `THINK_NEED` steps (below that the emitted answer is
  mostly a uniform guess); thinking longer adds nothing. This mirrors real
  reasoning — some thinking genuinely helps, but *overlong* chains are wasted —
  so the optimum is a small positive length. Without this, correctness-only RL
  would trivially drive the length to zero, misrepresenting the phenomenon.
- **Warm-start then length RL.** Like Kimi (SFT → RL), we first warm-start the
  reasoner at full reveal, then RL optimizes only the thinking budget. This keeps
  the exploration noise of short chains from eroding the trained reasoner and
  isolates the length-control effect.

The length-reward formula, the group-relative (GRPO-style) advantage, and the
"keep accuracy, cut length toward the minimum needed" outcome all match the
paper's length-penalty / long2short description.
