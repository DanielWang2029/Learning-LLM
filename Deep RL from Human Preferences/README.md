# Deep Reinforcement Learning from Human Preferences

A faithful, minimal, fully self-contained reproduction of Christiano et al.,
*"Deep Reinforcement Learning from Human Preferences"* (2017),
arXiv:[1706.03741](https://arxiv.org/abs/1706.03741).

The paper's central idea: instead of hand-coding a reward, **learn one from
human preferences over pairs of trajectory segments**, then optimize a policy
against that learned reward. To keep everything reproducible on CPU, the human
labeler is *simulated* by a **hidden true reward** on a small gridworld — the
agent and the reward model never see this reward, only which of two segments
collected more of it.

```
Deep RL from Human Preferences/
├── deep_rl_from_human_preferences.pdf   # the paper itself
├── requirements.txt                     # pinned CPU dependencies (torch, numpy)
├── src/                                 # the method, from scratch
│   ├── gridworld.py                     #   env with HIDDEN reward + value iteration
│   ├── preferences.py                   #   segment sampling + preference labeling (the "human")
│   └── reward_model.py                  #   Eq.1  reward net r̂ + Bradley-Terry loss
├── data/                                # results JSON written by the demo (generated)
├── demo/run_demo.py                     # learn reward from preferences, then plan a policy
└── visualization/index.html            # interactive loop, reward heatmaps, learning curve
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

The pipeline:

1. **Reference points** — compute the optimal policy (planning on the *true*
   reward) and a random policy, to bracket performance.
2. **Collect preferences** — sample random trajectory segments and label each
   pair by the hidden true reward (the simulated human).
3. **Train the reward model** — fit `r̂` with the **Bradley-Terry** logistic
   loss so preferred segments get higher predicted return.
4. **Plan against `r̂`** — run value iteration on the *learned* reward and
   measure that policy's **true** return.

## 3. Expected output

Seeded and reproducible. A representative run (≈2 s on CPU):

```
reward-model held-out preference accuracy : 100.0%
true return — random policy               : -5.911
true return — policy from LEARNED reward   :  5.243
true return — OPTIMAL (from true reward)   :  5.243
optimality gap closed vs random           : 100.0%

OK: reward learned from preferences yields a near-optimal policy.
```

The proof it works: the reward model predicts held-out preferences with high
accuracy, and the policy optimized purely on the *learned* reward achieves
essentially the **same true return as the optimal policy** — even though it
never observed the true reward. The demo `assert`s both.

Why it works despite the reward being learned only "up to a constant": equal-
length segment comparisons pin down the reward only up to a positive scale and
an offset, and the optimal policy of a discounted infinite-horizon MDP is
invariant to precisely that transform.

## 4. Explore the visualization

Open `visualization/index.html` in any browser (offline via `file://`). It
shows the preference-learning loop, side-by-side heatmaps of the **true** vs
**learned** reward with the greedy policy arrows and start→goal path, the
per-policy true returns, and the learning curve — all baked in from your demo
run. Serve the folder to load live data:

```bash
python -m http.server 8000   # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper concept | Component | File |
|---|---|---|
| Environment & agent | Gridworld, transitions, value iteration | `src/gridworld.py` |
| Segments & human feedback | Segment sampling, preference labels | `src/preferences.py` |
| Reward model r̂ | Reward network over observations | `src/reward_model.py` → `RewardModel` |
| Eq. 1 — preference predictor | Bradley-Terry logistic loss | `src/reward_model.py` → `preference_loss` |
| Policy optimization | Plan on learned reward, evaluate true return | `demo/run_demo.py` |
