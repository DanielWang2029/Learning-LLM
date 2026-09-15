"""End-to-end Language World Model demo for Qwen-AgentWorld (CPU, < 60s).

We (1) train a world model to predict next-state from (state, action) on a toy
gridworld, holding out transitions to measure generalization; (2) use the
learned model to PLAN a route to a goal *entirely in imagination* (never calling
the real environment); and (3) execute that plan in the real environment to show
the imagined trajectory matches reality and reaches the goal.

Run with:  python demo/run_demo.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))

from src import GridWorld, WorldModel, plan  # noqa: E402
from src.env import ACTION_NAMES  # noqa: E402
from src.world_model import accuracy, loss_fn  # noqa: E402


def load_env():
    path = ROOT / "data" / "env.json"
    if not path.exists():
        import generate_data  # noqa: E402
        generate_data.main()
    cfg = json.loads(path.read_text())
    env = GridWorld(cfg["size"], set(cfg["walls"]))
    return env, cfg


def rollout_real(env, start, actions):
    states = [start]
    s = start
    for a in actions:
        s = env.step(s, a)
        states.append(s)
    return states


def rollout_imagined(model, start, actions):
    states = [start]
    s = start
    for a in actions:
        s = model.predict(s, a)
        states.append(s)
    return states


def main() -> None:
    torch.manual_seed(0)
    t0 = time.time()
    env, cfg = load_env()

    trans = env.transitions()
    S = torch.tensor([t[0] for t in trans])
    A = torch.tensor([t[1] for t in trans])
    NS = torch.tensor([t[2] for t in trans])

    # Hold out 15% of transitions to measure that the model *generalizes*
    # the environment dynamics rather than memorizing every pair.
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(trans), generator=g)
    n_test = max(1, int(0.15 * len(trans)))
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    model = WorldModel(env.size, d_model=64)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)

    print(f"Env: {env.size}x{env.size} grid, {len(env.walls)} walls | "
          f"{len(trans)} transitions ({len(train_idx)} train / {n_test} test)\n")

    for step in range(1, 801):
        model.train()
        loss = loss_fn(model, S[train_idx], A[train_idx], NS[train_idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 200 == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                acc_tr = accuracy(model, S[train_idx], A[train_idx], NS[train_idx])
                acc_te = accuracy(model, S[test_idx], A[test_idx], NS[test_idx])
            print(f"step {step:3d} | loss {loss.item():.4f} | "
                  f"train acc {acc_tr*100:5.1f}% | held-out acc {acc_te*100:5.1f}%")

    model.eval()
    with torch.no_grad():
        acc_all = accuracy(model, S, A, NS)
        acc_test = accuracy(model, S[test_idx], A[test_idx], NS[test_idx])

    # --- Plan the main route in imagination, then verify in reality. ---
    start, goal = cfg["start"], cfg["goal"]
    actions = plan(model, start, goal)
    imagined = rollout_imagined(model, start, actions) if actions is not None else []
    real = rollout_real(env, start, actions) if actions is not None else []
    plan_ok = bool(actions is not None and real and real[-1] == goal)
    imagined_matches_real = imagined == real

    print("\n=== Planning in imagination ===")
    print(f"start={start}  goal={goal}")
    if actions is not None:
        print(f"plan (imagined): {[ACTION_NAMES[a] for a in actions]}")
        print(f"imagined states: {imagined}")
        print(f"real states    : {real}")
        print(f"reaches goal in real env: {plan_ok} | "
              f"imagined == real: {imagined_matches_real}")

    # --- Aggregate planning success over many start/goal pairs. ---
    free = [c for c in range(env.n_cells) if c not in env.walls]
    rng = torch.Generator().manual_seed(7)
    pairs, success = 0, 0
    for _ in range(60):
        i = free[int(torch.randint(0, len(free), (1,), generator=rng))]
        j = free[int(torch.randint(0, len(free), (1,), generator=rng))]
        if i == j:
            continue
        pairs += 1
        acts = plan(model, i, j)
        if acts is not None and rollout_real(env, i, acts)[-1] == j:
            success += 1
    plan_rate = success / pairs

    print("\n=== Results ===")
    print(f"next-state accuracy (all)      : {acc_all*100:.1f}%")
    print(f"next-state accuracy (held-out) : {acc_test*100:.1f}%")
    print(f"imagined plan reaches goal      : {plan_ok}")
    print(f"planning success over {pairs} pairs : {plan_rate*100:.1f}%")
    print(f"elapsed: {time.time()-t0:.1f}s")

    out = {
        "size": env.size,
        "walls": sorted(env.walls),
        "start": start,
        "goal": goal,
        "acc_all": acc_all,
        "acc_held_out": acc_test,
        "plan_actions": actions if actions is not None else [],
        "plan_action_names": [ACTION_NAMES[a] for a in actions] if actions else [],
        "imagined_states": imagined,
        "real_states": real,
        "imagined_matches_real": imagined_matches_real,
        "plan_reaches_goal": plan_ok,
        "planning_success_rate": plan_rate,
    }
    art = ROOT / "data" / "world_model_result.json"
    art.write_text(json.dumps(out))
    print(f"\nWrote {art}")

    assert acc_test > 0.85, "world model failed to generalize the dynamics"
    assert plan_ok and imagined_matches_real, "imagined plan did not match reality"
    assert plan_rate > 0.9, "planning in imagination unreliable"
    print("OK: learned world model predicts next states and plans in imagination.")


if __name__ == "__main__":
    main()
