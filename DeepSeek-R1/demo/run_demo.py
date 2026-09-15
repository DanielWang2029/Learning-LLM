"""End-to-end DeepSeek-R1 (GRPO) demo (CPU, a few seconds).

Trains a small parameterized policy *from scratch with reinforcement learning
only* to solve ``a+b`` while emitting a reasoning format ``{a+b}[answer]`` — a
stand-in for ``<think>…</think><answer>…</answer>``. There are NO supervised
targets, only a scalar reward for (format + showing work + correct answer),
optimized with GRPO:

    sample a GROUP of outputs per prompt
    advantage_i = (reward_i - mean) / std        # group-relative, no value net
    loss = - mean( advantage_i * logπ(output_i) ) - β · entropy

Watch reward, answer-accuracy, and format/think usage all climb from chance,
and the reasoning format emerge. Results go to ``data/r1_results.json``.

Run with:  python demo/run_demo.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(1)  # shared 4-core box: avoid thread oversubscription

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import (ParametrizedReasoner, all_prompts, compose, group_advantages,
                 grpo_loss, reward)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@torch.no_grad()
def evaluate(policy, prompts, device):
    """Greedy decode every prompt; return (accuracy, format-rate, think-rate, mean reward)."""
    idx = torch.arange(len(prompts), device=device)
    u, t, d = policy.greedy(idx)
    acc = fmt = think = tot = 0.0
    samples = []
    for i, (a, b) in enumerate(prompts):
        s = compose(int(u[i]), int(t[i]), int(d[i]), a, b)
        rd = reward(a, b, s)
        acc += rd["answer_ok"]; fmt += rd["format_ok"]; think += rd["think_ok"]; tot += rd["reward"]
        samples.append((f"{a}+{b}={s}", rd))
    n = len(prompts)
    return acc / n, fmt / n, think / n, tot / n, samples


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--group-size", type=int, default=12)
    p.add_argument("--max-operand", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--entropy-coef", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cpu")
    torch.manual_seed(args.seed)
    prompts = all_prompts(args.max_operand)
    policy = ParametrizedReasoner(len(prompts)).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    gen = torch.Generator().manual_seed(args.seed)
    G = args.group_size

    print(f"prompts={len(prompts)} (a,b in 0..{args.max_operand}) | group size G={G} | "
          f"GRPO (group-relative advantage, no value network)\n")

    # Repeat each prompt index G times -> one big group batch.
    idx_base = torch.arange(len(prompts), device=device)
    idx_rep = idx_base.repeat_interleave(G)

    # Step 0: chance-level baseline before any RL update.
    acc0, fmt0, think0, mr0, _ = evaluate(policy, prompts, device)
    curve = [{"step": 0, "acc": round(acc0, 4), "format": round(fmt0, 4),
              "think": round(think0, 4), "reward": round(mr0, 4)}]
    print(f"  step    0/{args.steps} (chance) | acc {acc0*100:5.1f}% | "
          f"format {fmt0*100:5.1f}% | think {think0*100:5.1f}%")

    start = time.time()
    for step in range(1, args.steps + 1):
        u, t, d = policy.sample(idx_rep, generator=gen)
        # Rewards for every sampled response.
        rewards = torch.empty(len(prompts) * G, device=device)
        for r in range(idx_rep.numel()):
            a, b = prompts[int(idx_rep[r])]
            s = compose(int(u[r]), int(t[r]), int(d[r]), a, b)
            rewards[r] = reward(a, b, s)["reward"]
        rewards_g = rewards.view(len(prompts), G)
        adv = group_advantages(rewards_g).view(-1)
        logp = policy.logprob(idx_rep, u, t, d)
        loss = grpo_loss(logp, adv) - args.entropy_coef * policy.entropy()
        opt.zero_grad(); loss.backward(); opt.step()

        if step % 15 == 0 or step == 1:
            acc, fmt, think, mr, _ = evaluate(policy, prompts, device)
            curve.append({"step": step, "acc": round(acc, 4), "format": round(fmt, 4),
                          "think": round(think, 4), "reward": round(mr, 4)})
            print(f"  step {step:4d}/{args.steps} | mean group reward {rewards.mean():.3f} "
                  f"| acc {acc*100:5.1f}% | format {fmt*100:5.1f}% | think {think*100:5.1f}%")

    elapsed = time.time() - start
    acc, fmt, think, mr, samples = evaluate(policy, prompts, device)
    print(f"\nTrained {args.steps} GRPO steps in {elapsed:.1f}s")
    print(f"Final: accuracy {acc*100:.1f}% | format-usage {fmt*100:.1f}% | "
          f"think-usage {think*100:.1f}% | mean reward {mr:.3f}\n")
    print("Sample greedy generations (prompt = response):")
    for text, rd in samples[:8]:
        flags = ("fmt" if rd["format_ok"] else "   ") + " " + \
                ("think" if rd["think_ok"] else "     ") + " " + \
                ("ans" if rd["answer_ok"] else "   ")
        print(f"  {text:16s}  [{flags}]  reward {rd['reward']:.1f}")

    examples = []
    for a, b in [(1, 2), (3, 4), (4, 4), (2, 0)]:
        i = prompts.index((a, b))
        u, t, d = policy.greedy(torch.tensor([i]))
        s = compose(int(u[0]), int(t[0]), int(d[0]), a, b)
        rd = reward(a, b, s)
        examples.append({"prompt": f"{a}+{b}=", "response": s, "answer": str(a + b),
                         "correct": rd["answer_ok"], "format_ok": rd["format_ok"]})

    out = {
        "config": {"steps": args.steps, "group_size": G, "max_operand": args.max_operand,
                   "n_prompts": len(prompts), "seconds": round(elapsed, 1)},
        "final": {"accuracy": round(acc, 4), "format": round(fmt, 4),
                  "think": round(think, 4), "reward": round(mr, 4)},
        "curve": curve,
        "examples": examples,
    }
    DATA_DIR.mkdir(exist_ok=True)
    out_path = DATA_DIR / "r1_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")

    if acc < 0.9 or fmt < 0.9:
        raise SystemExit(f"GRPO did not converge (acc {acc*100:.0f}%, format {fmt*100:.0f}%).")
    print("\nOK: from reward alone, the policy learned the reasoning format AND correct answers.")


if __name__ == "__main__":
    main()
