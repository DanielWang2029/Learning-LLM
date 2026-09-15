"""Learn a reward from preferences, then optimize a policy against it.

Reproduces the core loop of Christiano et al. (2017) on a 5×5 gridworld with a
HIDDEN true reward:

  1. sample trajectory segments (a random walk),
  2. label preferences with the hidden true reward (the "human"),
  3. fit a reward model r̂ with the Bradley-Terry loss,
  4. plan a policy against r̂ (value iteration), and
  5. measure that policy's TRUE return.

Success = the policy trained on the *learned* reward achieves nearly the same
true return as the optimal policy (which is planned on the true reward) —
i.e. preference-based reward learning recovered what to do without ever seeing
the reward. Runs on CPU in a few seconds.

Run with:  python demo/run_demo.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
torch.set_num_threads(1)

from src.gridworld import (
    ACTION_NAMES,
    GridWorld,
    evaluate_true_return,
    rollout_states,
    value_iteration,
)
from src.preferences import (
    make_preference_dataset,
    random_segment,
    segment_true_return,
)
from src.reward_model import RewardModel, preference_loss

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SEED = 0


def banner(t):
    print("\n" + "=" * 66 + f"\n{t}\n" + "=" * 66)


def main() -> None:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    t0 = time.time()

    env = GridWorld(n=5, gamma=0.9)
    print("Deep RL from Human Preferences — miniature gridworld")
    print(f"  {env.n}x{env.n} grid | start={env.s_to_rc(env.start)} "
          f"goal={env.s_to_rc(env.goal)} (hidden reward peaks at the goal)")

    # --- Reference points: optimal (plan on TRUE reward) and random ---------
    optimal_policy = value_iteration(env, env.true_reward)
    optimal_return = evaluate_true_return(env, optimal_policy)
    random_policy = rng.integers(0, len(ACTION_NAMES), size=env.num_states)
    random_return = evaluate_true_return(env, random_policy)
    print(f"  optimal true return (plans on hidden reward): {optimal_return:.3f}")
    print(f"  random-policy true return                   : {random_return:.3f}")

    # --- Stage: collect preferences over segments ---------------------------
    banner("Collect preference comparisons (labeled by the hidden reward)")
    SEG_LEN, N_PAIRS = 8, 600
    seg_a, seg_b, labels = make_preference_dataset(env, N_PAIRS, SEG_LEN, rng)
    ha, hb, hlab = make_preference_dataset(env, 200, SEG_LEN, rng)  # held-out
    print(f"  collected {len(labels)} training + {len(hlab)} held-out comparisons "
          f"(segment length {SEG_LEN})")

    A = torch.tensor(seg_a, dtype=torch.long)
    B = torch.tensor(seg_b, dtype=torch.long)
    L = torch.tensor(labels)
    HA = torch.tensor(ha, dtype=torch.long)
    HB = torch.tensor(hb, dtype=torch.long)
    HL = torch.tensor(hlab)

    # --- Stage: train the reward model --------------------------------------
    banner("Train reward model r̂ from preferences (Bradley-Terry)")
    rm = RewardModel(env.num_states)
    opt = torch.optim.Adam(rm.parameters(), lr=0.01, weight_decay=1e-4)
    EPOCHS = 120
    curve = []

    def held_out_acc():
        with torch.no_grad():
            ra = rm.state_reward(HA).sum(1)
            rb = rm.state_reward(HB).sum(1)
            pred = (ra > rb).float()
            return ((pred == HL).float().mean().item())

    def learned_policy_return():
        field = rm.reward_field().numpy()
        pol = value_iteration(env, field)
        return evaluate_true_return(env, pol), pol

    for epoch in range(1, EPOCHS + 1):
        ret_a = rm.state_reward(A).sum(1)
        ret_b = rm.state_reward(B).sum(1)
        loss = preference_loss(ret_a, ret_b, L)
        opt.zero_grad(); loss.backward(); opt.step()

        if epoch % 10 == 0 or epoch == 1:
            acc = held_out_acc()
            lr_ret, _ = learned_policy_return()
            curve.append({"epoch": epoch, "pref_acc": acc, "true_return": lr_ret})
            print(f"  epoch {epoch:3d}/{EPOCHS} | bt loss {loss.item():.4f} | "
                  f"held-out pref-acc {acc*100:5.1f}% | learned-policy true return {lr_ret:.3f}")

    final_acc = held_out_acc()
    learned_return, learned_policy = learned_policy_return()

    # --- Results ------------------------------------------------------------
    banner("RESULTS")
    print(f"  reward-model held-out preference accuracy : {final_acc*100:5.1f}%")
    print(f"  true return — random policy               : {random_return:.3f}")
    print(f"  true return — policy from LEARNED reward   : {learned_return:.3f}")
    print(f"  true return — OPTIMAL (from true reward)   : {optimal_return:.3f}")
    frac = (learned_return - random_return) / (optimal_return - random_return + 1e-9)
    print(f"  optimality gap closed vs random           : {frac*100:5.1f}%")

    # concrete: two segments and the preference, with the model's prediction
    a = random_segment(env, SEG_LEN, rng)
    b = random_segment(env, SEG_LEN, rng)
    ra_t, rb_t = segment_true_return(env, a), segment_true_return(env, b)
    with torch.no_grad():
        ra_h = rm.segment_return(a).item()
        rb_h = rm.segment_return(b).item()
    print("\n  example comparison:")
    print(f"    segment A cells {[env.s_to_rc(s) for s in a]}  true={ra_t:.2f} r̂={ra_h:.2f}")
    print(f"    segment B cells {[env.s_to_rc(s) for s in b]}  true={rb_t:.2f} r̂={rb_h:.2f}")
    print(f"    human prefers {'A' if ra_t>rb_t else 'B'}; model prefers "
          f"{'A' if ra_h>rb_h else 'B'}")

    # --- Persist for the visualization --------------------------------------
    learned_field = rm.reward_field().numpy()
    DATA_DIR.mkdir(exist_ok=True)
    payload = {
        "n": env.n,
        "start": env.start, "goal": env.goal,
        "true_reward": env.true_reward.tolist(),
        "learned_reward": learned_field.tolist(),
        "optimal_return": optimal_return,
        "random_return": random_return,
        "learned_return": learned_return,
        "pref_accuracy": final_acc,
        "learned_policy": [ACTION_NAMES[a] for a in learned_policy.tolist()],
        "optimal_policy": [ACTION_NAMES[a] for a in optimal_policy.tolist()],
        "learned_path": rollout_states(env, learned_policy),
        "optimal_path": rollout_states(env, optimal_policy),
        "learning_curve": curve,
        "n_pairs": len(labels), "seg_len": SEG_LEN,
    }
    out = DATA_DIR / "preference_results.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n  wrote {out}")
    print(f"  total time: {time.time()-t0:.1f}s")

    assert final_acc > 0.9, f"reward model preference accuracy too low: {final_acc:.2f}"
    assert learned_return > 0.9 * optimal_return, (
        f"learned-reward policy underperforms: {learned_return:.3f} vs optimal {optimal_return:.3f}"
    )
    print("\nOK: reward learned from preferences yields a near-optimal policy.")


if __name__ == "__main__":
    main()
