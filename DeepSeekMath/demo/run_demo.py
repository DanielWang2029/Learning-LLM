"""End-to-end GRPO demo (CPU, well under a minute) — DeepSeekMath §4.1.

A tiny Transformer policy learns to add, *from reward alone*, using Group
Relative Policy Optimization. There are NO supervised answer labels and NO value
network. Each step:

  1. sample a GROUP of G completions per prompt "a + b ="            (exploration)
  2. score each with the outcome reward (1.0 if the answer is correct)
  3. standardize rewards WITHIN each group -> advantages              (Eq. 20)
       A_i = (R_i - mean) / (std + eps)
  4. GRPO update: clipped policy-gradient surrogate minus a KL        (Eq. 21)
     penalty to a frozen reference model.

Watch mean reward and greedy answer-accuracy climb from chance to ~100%. The
demo also prints one worked group so you can see the group-relative advantage
being computed. Results go to ``data/grpo_results.json`` for the viz.

Run with:  python demo/run_demo.py
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(1)

PAPER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PAPER_DIR))

from src import (EOS, PolicyConfig, TransformerPolicy, all_prompts, decode_answer,
                 encode_prompt, group_advantages, grpo_loss, max_answer_len, reward)  # noqa: E402

DATA_DIR = PAPER_DIR / "data"


def build_prompt_tensor(prompts, device):
    return torch.tensor([encode_prompt(a, b) for a, b in prompts], device=device)


@torch.no_grad()
def evaluate(policy, prompts, gen_len, device):
    """Greedy-decode every prompt; return (accuracy, mean reward, samples)."""
    correct = total_r = 0.0
    samples = []
    for a, b in prompts:
        gen = policy.greedy(encode_prompt(a, b), gen_len, EOS)
        ans = decode_answer(gen)
        r = reward(a, b, ans)
        correct += (ans == str(a + b))
        total_r += r
        samples.append((a, b, ans, r))
    n = len(prompts)
    return correct / n, total_r / n, samples


def worked_group(policy, prompts, G, gen_len, device, gen):
    """Compute and print the group-relative advantages for one prompt.

    Returns a dict describing the group so the visualization can show real data.
    """
    a, b = prompts[len(prompts) // 2]
    prompt = torch.tensor(encode_prompt(a, b), device=device)[None].repeat(G, 1)
    seq, plen = policy.sample_batch(prompt, gen_len, EOS, gen)
    rewards = torch.tensor([reward(a, b, decode_answer(seq[i, plen:].tolist()))
                            for i in range(G)])
    adv = group_advantages(rewards[None]).squeeze(0)
    print(f"\nWorked group for prompt  {a} + {b} = {a + b}   (G={G}):")
    print(f"  {'sample':>18} | {'reward':>6} | {'advantage':>9}")
    members = []
    for i in range(G):
        ans = decode_answer(seq[i, plen:].tolist())
        quoted = '"' + ans + '"'
        print(f"  {quoted:>18} | {rewards[i].item():>6.1f} | {adv[i].item():>+9.3f}")
        members.append({"answer": ans, "reward": round(float(rewards[i]), 3),
                        "advantage": round(float(adv[i]), 3)})
    print(f"  group mean reward = {rewards.mean():.3f}, std = {rewards.std():.3f}  "
          f"->  A_i = (R_i - mean) / (std + eps)")
    return {"prompt": f"{a} + {b} =", "target": a + b, "group_size": G,
            "mean": round(float(rewards.mean()), 3), "std": round(float(rewards.std()), 3),
            "members": members}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--max-operand", type=int, default=5)
    p.add_argument("--inner-epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--beta", type=float, default=0.02)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--ref-update-every", type=int, default=0,
                   help="refresh the KL reference to the current policy (Algorithm 1 outer loop); 0 disables")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cpu")
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    prompts = all_prompts(args.max_operand)
    gen_len = max_answer_len(args.max_operand) + 1  # +1 for EOS
    cfg = PolicyConfig(dim=48, n_heads=4, n_layers=2, max_len=16)
    policy = TransformerPolicy(cfg).to(device)
    reference = copy.deepcopy(policy).eval()  # frozen reference for the KL penalty
    for pm in reference.parameters():
        pm.requires_grad_(False)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    n_params = sum(pm.numel() for pm in policy.parameters())
    print("=" * 72)
    print("DeepSeekMath GRPO demo — arXiv:2402.03300 (origin of GRPO)")
    print("=" * 72)
    print(f"policy params={n_params:,} | prompts={len(prompts)} (a,b in 0..{args.max_operand}) "
          f"| group size G={args.group_size}")
    print("critic-free: NO value network | reward-only: NO answer labels\n")

    prompt_tensor = build_prompt_tensor(prompts, device)
    idx_rep = torch.arange(len(prompts)).repeat_interleave(args.group_size)
    prompts_rep = prompt_tensor[idx_rep]

    acc0, mr0, _ = evaluate(policy, prompts, gen_len, device)
    curve = [{"step": 0, "acc": round(acc0, 4), "reward": round(mr0, 4), "kl": 0.0}]
    print(f"  step   0/{args.steps} (chance) | mean reward {mr0:.3f} | accuracy {acc0*100:5.1f}%")

    start = time.time()
    for step in range(1, args.steps + 1):
        # --- 1. sample a group of completions per prompt (behavior policy) ---
        seq, plen = policy.sample_batch(prompts_rep, gen_len, EOS, gen, args.temperature)
        mask = _answer_mask(seq, plen, EOS)
        with torch.no_grad():
            logp_old = policy.token_logprobs(seq, plen)
            logp_ref = reference.token_logprobs(seq, plen)

        # --- 2. rewards + 3. group-relative advantages ---
        rewards = torch.tensor(
            [reward(*prompts[int(idx_rep[i])], decode_answer(seq[i, plen:].tolist()))
             for i in range(seq.size(0))], device=device)
        adv = group_advantages(rewards.view(len(prompts), args.group_size)).view(-1)

        # --- 4. GRPO update (a few inner epochs reuse the same group) ---
        for _ in range(args.inner_epochs):
            logp_new = policy.token_logprobs(seq, plen)
            loss, diag = grpo_loss(logp_new, logp_old, logp_ref, adv, mask,
                                   beta=args.beta, clip_eps=args.clip_eps)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()

        # GRPO Algorithm 1 refreshes the reference policy in its outer loop, so
        # the KL anchor tracks the improving model instead of the initial one.
        if args.ref_update_every and step % args.ref_update_every == 0:
            reference.load_state_dict(policy.state_dict())

        if step % 10 == 0 or step == 1:
            acc, mr, _ = evaluate(policy, prompts, gen_len, device)
            curve.append({"step": step, "acc": round(acc, 4), "reward": round(mr, 4),
                          "kl": round(diag["kl"], 4)})
            print(f"  step {step:3d}/{args.steps} | group reward {rewards.mean():.3f} "
                  f"| accuracy {acc*100:5.1f}% | KL {diag['kl']:.3f} | ratio {diag['ratio']:.3f}")

    elapsed = time.time() - start
    acc, mr, samples = evaluate(policy, prompts, gen_len, device)
    print(f"\nTrained {args.steps} GRPO steps in {elapsed:.1f}s")
    print(f"Final: accuracy {acc*100:.1f}% | mean reward {mr:.3f}")

    wg = worked_group(policy, prompts, args.group_size, gen_len, device, gen)

    print("\nSample greedy answers:")
    for a, b, ans, r in samples[:8]:
        ok = "OK " if ans == str(a + b) else "   "
        print(f"  {a} + {b} = {ans:>3}  (target {a + b:>2})  [{ok}] reward {r:.1f}")

    examples = [{"prompt": f"{a} + {b} =", "answer": ans, "target": str(a + b),
                 "correct": ans == str(a + b)} for a, b, ans, r in samples[:12]]
    out = {
        "paper": "DeepSeekMath (arXiv:2402.03300) — origin of GRPO",
        "config": {"steps": args.steps, "group_size": args.group_size,
                   "max_operand": args.max_operand, "n_prompts": len(prompts),
                   "beta": args.beta, "clip_eps": args.clip_eps,
                   "policy_params": n_params, "seconds": round(elapsed, 1)},
        "final": {"accuracy": round(acc, 4), "reward": round(mr, 4)},
        "curve": curve,
        "worked_group": wg,
        "examples": examples,
    }
    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "grpo_results.json").write_text(json.dumps(out, indent=2))
    print("\nWrote data/grpo_results.json (reward/accuracy curve for the viz).")

    if acc < 0.9:
        raise SystemExit(f"GRPO did not converge (accuracy {acc*100:.0f}% < 90%).")
    print("OK: from reward alone (no labels, no value network), GRPO taught the policy to add.")


def _answer_mask(seq: torch.Tensor, plen: int, eos_id: int) -> torch.Tensor:
    """1 for generated tokens up to & including the first EOS, else 0."""
    gen = seq[:, plen:]
    is_eos = (gen == eos_id)
    after_eos = is_eos.cumsum(dim=1) - is_eos.long()  # 0 until first EOS inclusive
    return (after_eos == 0).float()


if __name__ == "__main__":
    main()
