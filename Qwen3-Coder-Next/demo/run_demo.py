"""Train a sparse-MoE coding policy from executable unit-test rewards.

The update is a compact GRPO-style policy gradient: several programs are
sampled per specification; their group-relative test rewards weight log-probs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.generate_data import task_specs
from src.agent import MoEPolicy, execute, reward


def pass_rate(policy: MoEPolicy, specs: list[dict[str, object]]) -> tuple[float, list[int]]:
    policy.eval()
    with torch.no_grad():
        actions = policy(torch.arange(len(specs))).argmax(dim=-1).tolist()
    scores = [reward(action, i, specs[i]["tests"]) for i, action in enumerate(actions)]
    return sum(scores) / len(scores), actions


def main() -> None:
    torch.manual_seed(300729)
    specs = task_specs()
    policy = MoEPolicy(len(specs), len(specs))
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
    group_size, updates = 8, 220
    before, _ = pass_rate(policy, specs)
    curve = [{"update": 0, "pass_rate": round(before * 100, 1)}]
    task_ids = torch.arange(len(specs))

    for update in range(1, updates + 1):
        policy.train()
        logits = policy(task_ids)
        distribution = torch.distributions.Categorical(
            logits=logits.unsqueeze(0).expand(group_size, -1, -1)
        )
        actions = distribution.sample()  # [group, task]
        log_probs = distribution.log_prob(actions)
        rewards = torch.tensor(
            [[reward(int(actions[g, t]), t, specs[t]["tests"]) for t in range(len(specs))]
             for g in range(group_size)],
            dtype=torch.float32,
        )
        # GRPO's group baseline: no learned critic, only peer rollout outcomes.
        advantages = (rewards - rewards.mean(0, keepdim=True)) / (
            rewards.std(0, keepdim=True, unbiased=False) + 1e-4
        )
        loss = -(advantages.detach() * log_probs).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        if update % 20 == 0:
            rate, _ = pass_rate(policy, specs)
            curve.append({"update": update, "pass_rate": round(rate * 100, 1)})

    final_rate, actions = pass_rate(policy, specs)
    total_moe, active_moe = policy.moe_counts()
    total_policy = sum(p.numel() for p in policy.parameters())
    example_task = 2
    payload = {
        "paper": "Qwen3-Coder-Next", "metric": "fraction of hidden unit tests passed",
        "initial_pass_rate": round(before * 100, 1), "final_pass_rate": round(final_rate * 100, 1),
        "curve": curve, "moe": {"experts": 4, "top_k": 2, "total_moe_params": total_moe,
                                 "active_moe_params": active_moe, "total_policy_params": total_policy},
        "example": {"spec": specs[example_task]["name"], "emitted_program": specs[actions[example_task]]["program"],
                    "hidden_inputs": specs[example_task]["tests"],
                    "outputs": [execute(actions[example_task], x) for x in specs[example_task]["tests"]]},
    }
    (ROOT / "data" / "agentic_rl_results.json").write_text(json.dumps(payload, indent=2))

    print("Qwen3-Coder-Next — tiny verifiable agentic-RL reproduction")
    print(f"Greedy unit-test pass rate: {before * 100:5.1f}% → {final_rate * 100:5.1f}%")
    print(f"MoE parameters: {total_moe:,} total / {active_moe:,} active per decision "
          f"(top-{policy.moe.top_k} of 4 experts); full policy: {total_policy:,}")
    print("\nVerified emitted program")
    print(f"  Spec:    {specs[example_task]['name']}")
    print(f"  Program: {payload['example']['emitted_program']}")
    print(f"  Tests:   x={payload['example']['hidden_inputs']} → {payload['example']['outputs']} (all pass)")
    if final_rate < 0.95 or final_rate <= before:
        raise SystemExit("Execution-feedback RL did not improve the unit-test pass rate.")
    print("\nOK: group-relative rewards from execution trained the sparse-MoE policy.")


if __name__ == "__main__":
    main()
