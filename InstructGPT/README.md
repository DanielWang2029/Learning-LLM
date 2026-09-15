# InstructGPT — Training LMs to Follow Instructions with Human Feedback

A faithful, minimal, fully self-contained reproduction of the **three-stage
RLHF pipeline** from Ouyang et al., *"Training language models to follow
instructions with human feedback"* (2022), arXiv:[2203.02155](https://arxiv.org/abs/2203.02155).

Everything needed to read, run, and understand the method lives in **this
folder**: the paper PDF, a from-scratch PyTorch implementation, a runnable
end-to-end demo, and an interactive visualization.

Because using a real LLM and real human labelers is neither reproducible nor
CPU-friendly, we shrink the problem to a **toy alignment task with a known
rule**: the "helpful" response to a prompt is the prompt's tokens **sorted in
ascending order**. That rule plays the role of the hidden human preference, so
everything is deterministic and runs on a laptop in ~20 seconds.

```
InstructGPT/
├── instructgpt.pdf          # the paper itself
├── requirements.txt         # pinned CPU dependencies (torch, numpy)
├── src/                     # the method, from scratch
│   ├── model.py             #   TinyLM: a tiny GPT-style policy (SFT model / RL policy)
│   ├── reward_model.py      #   §2.3  RewardModel + Bradley-Terry preference loss
│   ├── data.py              #   the toy sort task: demonstrations & preference pairs
│   └── rlhf.py              #   §2.3  best-of-n rejection sampling + REINFORCE helpers
├── data/                    # results JSON written by the demo (generated)
├── demo/run_demo.py         # runs all 3 stages end-to-end and prints evidence
└── visualization/index.html # interactive SFT → RM → RL diagram + live metrics
```

## 1. Set up the environment

Requires Python 3.10+. This folder reuses the shared virtual environment that
ships with the reference reproduction:

```bash
source "/workspace/Attention Is All You Need/.venv/bin/activate"
```

To build a fresh one instead:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

This installs the CPU build of PyTorch — no GPU required.

## 2. Run the demo

```bash
python demo/run_demo.py
```

It executes the full pipeline and prints progress for every stage:

- **Stage 1 — SFT.** A `TinyLM` is supervised-fine-tuned on *imperfect*
  demonstrations (only ~55% are the ideal sorted answer, mimicking noisy human
  data). This yields a decent-but-flawed baseline policy.
- **Stage 2 — Reward Model.** A small reward model is trained on **preference
  pairs** (chosen vs rejected, labeled by the rule) with the Bradley-Terry
  logistic loss. It reports held-out preference accuracy.
- **Stage 3 — RLHF.** The policy is optimized against the reward model with a
  **REINFORCE** step (KL-penalized toward the SFT policy) and evaluated with
  **best-of-n** rejection sampling.

Finally it writes `data/rlhf_results.json` for the visualization.

## 3. Expected output

The exact numbers are seeded and reproducible. A representative run:

```
Reward-model preference accuracy      :  99.2%
SFT   policy — avg reward (sampled)   : 0.753 | compliance 30.9%
RLHF  policy — avg reward (sampled)   : 0.996 | compliance 68.0%
RLHF + best-of-8 (RM rejection samp.) : 1.000 | compliance 47.7%

OK: RM is accurate and RLHF increased rule-compliance over SFT.
```

The two things that prove the method works:

1. the **reward model is accurate** (>99% preference agreement), and
2. the **RLHF policy's average reward and rule-compliance clearly exceed the
   SFT baseline** — the policy learns to sort even though its own SFT data was
   only partly sorted, exactly the InstructGPT effect. The demo `assert`s both.

## 4. Explore the visualization

Open `visualization/index.html` in any browser (works offline via `file://`).
It shows the clickable **SFT → Reward Model → RL** pipeline, the Bradley-Terry
and REINFORCE equations, the reward-vs-training curve, and the final
SFT/RLHF/best-of-n metrics — all driven by the numbers your demo just produced
(baked into the page). Serve the folder to load live data instead:

```bash
python -m http.server 8000   # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper section | Component | File |
|---|---|---|
| §2.3, Step 1 | Supervised fine-tuning (SFT) | `src/model.py`, `demo/run_demo.py` |
| §2.3, Step 2 | Reward model | `src/reward_model.py` → `RewardModel` |
| §2.3, Step 2 | Bradley-Terry preference loss | `src/reward_model.py` → `bradley_terry_loss` |
| §2.3, Step 3 | RL policy optimization (REINFORCE + KL) | `demo/run_demo.py` (Stage 3) |
| §2.3, Step 3 | best-of-n rejection sampling | `src/rlhf.py` → `best_of_n` |
| Appendix | Preference data collection | `src/data.py` → `make_preference_pairs` |
