# DeepSeekMath — the origin of GRPO

A faithful, minimal, and fully self-contained reproduction of **Group Relative
Policy Optimization (GRPO)**, the reinforcement-learning algorithm introduced in
*"DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language
Models"* (2024, arXiv:2402.03300, §4.1). Everything needed to read, run, and
understand the algorithm lives in **this folder**.

GRPO is the RL method behind DeepSeek-R1 and much of the recent reasoning-RL
wave. Its key idea is to **delete PPO's value network** and instead use a *group*
of sampled outputs as the baseline:

- sample a **group** of `G` completions per prompt,
- standardize their rewards within the group to get advantages
  `A_i = (R_i − mean) / (std + eps)` — above-average = positive, below = negative,
- update with the PPO-style **clipped surrogate** plus a **KL penalty** toward a
  frozen reference model. No critic, ever.

> **Sibling note.** `DeepSeek-R1/` already uses GRPO-style RL to elicit a
> reasoning *format*. This folder deliberately focuses on the **GRPO algorithm
> and its derivation** — the group-relative advantage, the clipped ratio, and the
> KL term — with a genuine autoregressive policy trained by reward alone.

```
DeepSeekMath/
├── deepseekmath.pdf            # the paper itself
├── requirements.txt            # pinned CPU dependencies (torch + numpy)
├── src/
│   ├── grpo.py                 #   §4.1  advantages (Eq. 20) + clipped loss + KL (Eq. 21)
│   ├── policy.py               #   §4.1  tiny autoregressive Transformer policy
│   └── task.py                 #   §4    arithmetic task + outcome reward
├── data/
│   └── generate_data.py        # writes data/task.json (task + reward description)
├── demo/
│   └── run_demo.py             # trains the policy from reward alone with GRPO
└── visualization/
    └── index.html              # GRPO loop diagram + reward/accuracy curve
```

## 1. Set up the environment

Requires Python 3.10+. From inside this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## 2. Run the demo

```bash
python data/generate_data.py   # optional: writes data/task.json
python demo/run_demo.py        # ~17s on CPU
```

A tiny Transformer policy learns single-digit addition **purely from reward** —
no supervised answer labels and no value network. Each step samples a group of
completions per prompt, standardizes the rewards into advantages, and applies the
GRPO update.

## 3. Expected output

Accuracy and mean reward climb from chance to **~92%**, and the demo prints a
**worked group** so you can watch the group-relative advantage being computed:

```
Worked group for prompt  3 + 0 = 3   (G=8):
              sample | reward | advantage
                 "3" |    1.0 |    +0.353
                 ...
                "3?" |    0.0 |    -2.474
  group mean reward = 0.875, std = 0.354  ->  A_i = (R_i - mean) / (std + eps)
```

The correct completions receive a positive advantage and are made more likely;
the malformed one receives a large negative advantage and is suppressed — all
without a critic. Results are written to `data/grpo_results.json` for the viz.

## 4. Explore the visualization

Open `visualization/index.html` in any browser for an animated GRPO loop (group
sampling → reward → normalized advantage → clipped+KL update), the worked-group
advantage table, and the live reward/accuracy curve from your run. Works from
`file://`; serve over HTTP to load fresh JSON:

```bash
python -m http.server 8000     # then open http://localhost:8000/visualization/
```

## The GRPO update, precisely

```
A_i          = (R_i − mean(R_1..R_G)) / (std(R_1..R_G) + eps)        # Eq. 20
r_{i,t}      = π_θ(o_{i,t}) / π_θ_old(o_{i,t})                        # importance ratio
J            = mean_i mean_t [ min(r·A_i, clip(r,1±ε)·A_i)           # Eq. 21
                               − β · D_KL(π_θ ‖ π_ref) ]
D_KL         ≈ π_ref/π_θ − log(π_ref/π_θ) − 1                        # unbiased "k3" estimator
```

## Code ↔ paper map

| Paper section | Component | File |
|---|---|---|
| §4.1, Eq. 20 | Group-relative advantage | `src/grpo.py` → `group_advantages` |
| §4.1, Eq. 21 | Clipped surrogate + KL penalty | `src/grpo.py` → `grpo_loss` |
| §4.1 | Unbiased KL-to-reference estimator | `src/grpo.py` → `kl_to_reference` |
| §4.1 | Autoregressive policy (sampling / log-probs) | `src/policy.py` |
| §4 | Math task + rule-based reward | `src/task.py` |
