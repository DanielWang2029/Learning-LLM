# Direct Preference Optimization (DPO)

A faithful, minimal, fully self-contained reproduction of Rafailov et al.,
*"Direct Preference Optimization: Your Language Model is Secretly a Reward
Model"* (2023), arXiv:[2305.18290](https://arxiv.org/abs/2305.18290).

DPO aligns a language model to human preferences **without a reward model and
without reinforcement learning**. Its insight: the RLHF objective (maximize a
reward with a KL leash to a reference policy) has a closed-form optimum, which
lets you express the reward directly in terms of the policy. Substituting that
into the Bradley-Terry preference model gives a single supervised loss on
preference pairs.

To keep everything reproducible on CPU, the human preference is a **known
rule**: the preferred response to a prompt is its tokens **sorted in ascending
order**. We train a tiny reference LM on (imperfect) demonstrations, then run
DPO on `(chosen ≻ rejected)` pairs and watch the policy become aligned.

```
Direct Preference Optimization/
├── direct_preference_optimization.pdf   # the paper itself
├── requirements.txt                     # pinned CPU dependencies (torch, numpy)
├── src/                                 # the method, from scratch
│   ├── model.py                         #   TinyLM: shared arch for π_ref and π_θ
│   ├── data.py                          #   toy sort task: demonstrations & preference pairs
│   └── dpo.py                           #   Eq. 7  the DPO loss + implicit reward
├── data/                                # results JSON written by the demo (generated)
├── demo/run_demo.py                     # SFT reference + DPO end-to-end, with evidence
└── visualization/index.html            # RLHF↔DPO, the derivation, margin-over-training
```

## 1. Set up the environment

Requires Python 3.10+. Reuse the shared virtual environment:

```bash
source "/workspace/Attention Is All You Need/.venv/bin/activate"
```

Or build a fresh one:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## 2. Run the demo

```bash
python demo/run_demo.py
```

- **Stage 1 — SFT reference.** Train `TinyLM` on imperfect demonstrations and
  freeze a copy as `π_ref`.
- **Stage 2 — DPO.** Starting from `π_ref`, minimize the DPO loss on preference
  pairs. Frozen reference log-probs are precomputed once; each step is pure
  gradient descent — no sampling, no reward model.

## 3. Expected output

Seeded and reproducible. A representative run (≈8 s on CPU):

```
DPO loss                : 0.6931 (start) -> 0.0068 (end)
implicit reward margin  : +0.000 (start) -> +7.547 (end)
preference accuracy     : 99.2% of pairs have chosen ≻ rejected
π_ref  policy — avg reward (sampled): 0.753 | compliance 30.9%
π_DPO  policy — avg reward (sampled): 0.974 | compliance 74.6%

OK: DPO loss fell, reward margin grew, and generations became more preferred.
```

The three things that prove DPO works, all without a reward model or RL:

1. the **DPO loss decreases**,
2. the **implicit reward margin** (chosen minus rejected) **increases** and is
   positive on ~99% of pairs, and
3. the **policy's generations shift toward the preferred (sorted) behavior**,
   beating the SFT reference. The demo `assert`s all three.

## 4. Explore the visualization

Open `visualization/index.html` in any browser (offline via `file://`). It
contrasts the RLHF and DPO pipelines, walks through the derivation ("the LM is
secretly a reward model") and the Eq. 7 loss, and plots the **reward margin,
loss, and per-response implicit rewards over training** — all baked in from
your demo run. Serve the folder to load live data:

```bash
python -m http.server 8000   # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper section | Component | File |
|---|---|---|
| §4 / Eq. 7 | DPO loss | `src/dpo.py` → `dpo_loss` |
| §4 | Implicit reward β·(logπ_θ − logπ_ref) | `src/dpo.py` |
| §4 | Reference policy π_ref (frozen SFT) | `demo/run_demo.py` (Stage 1) |
| §4 | Per-response log π over the completion | `src/model.py` → `TinyLM.sequence_logprob` |
| §5 (setup) | Preference dataset (chosen ≻ rejected) | `src/data.py` → `make_preference_pairs` |
