"""End-to-end miniature RLHF pipeline (InstructGPT, Ouyang et al. 2022).

Runs the three stages on the toy "sort the tokens" task and prints clear
evidence that alignment worked:

  Stage 1  SFT  — supervised fine-tune a tiny LM on demonstrations.
  Stage 2  RM   — train a reward model on preference pairs (Bradley-Terry).
  Stage 3  RLHF — improve the policy against the RM via REINFORCE + best-of-n.

Success criteria (asserted at the end):
  * the reward model ranks chosen > rejected with high accuracy, and
  * the RLHF policy's average reward / rule-compliance beats the SFT baseline.

Runs on CPU in well under a minute.  Run with:  python demo/run_demo.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Tiny models on CPU run fastest single-threaded (and stay friendly on shared
# machines): the per-op threading overhead dominates at this scale.
torch.set_num_threads(1)

from src import data
from src.model import TinyLM
from src.reward_model import RewardModel, bradley_terry_loss

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SEED = 0


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def sft_loss(policy: TinyLM, full: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Teacher-forced cross-entropy over the response region only."""
    logits = policy(full[:, :-1])
    targets = full[:, 1:]
    m = mask[:, 1:]
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
    )
    loss = (loss * m.reshape(-1).float()).sum() / m.float().sum()
    return loss


@torch.no_grad()
def eval_policy(policy: TinyLM, prompts, temperature=1.0, greedy=False):
    """Average sortedness reward and exact-compliance over sampled responses.

    All prompts are decoded in a single batched call for CPU speed.
    """
    prefixes = torch.tensor([data.prefix_tokens(p) for p in prompts], dtype=torch.long)
    gen = policy.generate(prefixes, data.RESP_LEN, temperature=temperature, greedy=greedy)
    responses = gen[:, prefixes.size(1):].tolist()
    rewards = [data.sortedness(r) for r in responses]
    compliant = sum(data.is_compliant(p, r) for p, r in zip(prompts, responses))
    return sum(rewards) / len(rewards), compliant / len(prompts)


@torch.no_grad()
def best_of_n_eval(policy: TinyLM, rm: RewardModel, prompts, n: int):
    """best-of-n rejection sampling over all prompts at once.

    For each prompt draw n samples, keep the RM's top-scoring one, then measure
    its true rule reward / compliance.
    """
    flat = [p for p in prompts for _ in range(n)]
    prefixes = torch.tensor([data.prefix_tokens(p) for p in flat], dtype=torch.long)
    gen = policy.generate(prefixes, data.RESP_LEN, temperature=1.0)
    responses = gen[:, prefixes.size(1):].tolist()
    seqs = data.encode_batch(flat, responses)
    scores = rm(seqs).view(len(prompts), n)
    best = scores.argmax(dim=1)
    rewards, compliant = [], 0
    for i, p in enumerate(prompts):
        r = responses[i * n + int(best[i])]
        rewards.append(data.sortedness(r))
        compliant += int(data.is_compliant(p, r))
    return sum(rewards) / len(prompts), compliant / len(prompts)


def banner(title: str) -> None:
    print("\n" + "=" * 66)
    print(title)
    print("=" * 66)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    torch.manual_seed(SEED)
    rng = torch.Generator().manual_seed(SEED)
    t0 = time.time()

    print("InstructGPT in miniature — 3-stage RLHF on a toy sort task")
    print(
        f"vocab={data.VOCAB_SIZE}  prompt_len={data.PROMPT_LEN}  "
        f"rule: response must be the prompt sorted ascending"
    )

    # Fixed evaluation prompts, shared across all stages.
    eval_prompts = [data.random_prompt(rng) for _ in range(256)]
    response_mask = data.build_response_mask()

    # ---------------------------------------------------------------- Stage 1
    banner("STAGE 1 — Supervised Fine-Tuning (SFT) on demonstrations")
    policy = TinyLM(data.VOCAB_SIZE, data.SEQ_LEN)
    opt = torch.optim.Adam(policy.parameters(), lr=3e-3)
    p_dem, r_dem = data.make_demonstrations(2048, rng)
    full_dem = data.encode_batch(p_dem, r_dem)
    mask_b = response_mask.unsqueeze(0).expand(full_dem.size(0), -1)

    SFT_STEPS, BS = 260, 64
    for step in range(1, SFT_STEPS + 1):
        idx = torch.randint(0, full_dem.size(0), (BS,), generator=rng)
        loss = sft_loss(policy, full_dem[idx], mask_b[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 65 == 0 or step == 1:
            print(f"  step {step:3d}/{SFT_STEPS} | sft loss {loss.item():.4f}")

    sft_reward, sft_comp = eval_policy(policy, eval_prompts, temperature=1.0)
    sft_reward_g, sft_comp_g = eval_policy(policy, eval_prompts, greedy=True)
    print(f"\n  SFT policy (sampled @T=1): avg reward {sft_reward:.3f} | "
          f"exact-compliance {sft_comp*100:.1f}%")
    print(f"  SFT policy (greedy)      : avg reward {sft_reward_g:.3f} | "
          f"exact-compliance {sft_comp_g*100:.1f}%")

    # Snapshot the frozen SFT reference policy (used for the KL penalty).
    sft_ref = TinyLM(data.VOCAB_SIZE, data.SEQ_LEN)
    sft_ref.load_state_dict(policy.state_dict())
    for prm in sft_ref.parameters():
        prm.requires_grad_(False)

    # ---------------------------------------------------------------- Stage 2
    banner("STAGE 2 — Reward Model on preference pairs (Bradley-Terry)")
    rm = RewardModel(data.VOCAB_SIZE, data.SEQ_LEN)
    opt_rm = torch.optim.Adam(rm.parameters(), lr=3e-3)
    p_pref, chosen, rejected = data.make_preference_pairs(2048, rng)
    seq_ch = data.encode_batch(p_pref, chosen)
    seq_rj = data.encode_batch(p_pref, rejected)

    RM_STEPS = 260
    for step in range(1, RM_STEPS + 1):
        idx = torch.randint(0, seq_ch.size(0), (BS,), generator=rng)
        r_ch = rm(seq_ch[idx]); r_rj = rm(seq_rj[idx])
        loss = bradley_terry_loss(r_ch, r_rj)
        opt_rm.zero_grad(); loss.backward(); opt_rm.step()
        if step % 65 == 0 or step == 1:
            acc = (r_ch > r_rj).float().mean().item()
            print(f"  step {step:3d}/{RM_STEPS} | bt loss {loss.item():.4f} | "
                  f"batch pref-acc {acc*100:.1f}%")

    # Held-out preference accuracy.
    ph, ch_h, rj_h = data.make_preference_pairs(512, rng)
    with torch.no_grad():
        rc = rm(data.encode_batch(ph, ch_h))
        rr = rm(data.encode_batch(ph, rj_h))
    rm_acc = (rc > rr).float().mean().item()
    print(f"\n  Reward-model held-out preference accuracy: {rm_acc*100:.1f}%")

    # ---------------------------------------------------------------- Stage 3
    banner("STAGE 3 — RLHF: optimize policy vs RM (REINFORCE + KL penalty)")
    opt_pi = torch.optim.Adam(policy.parameters(), lr=1e-3)
    kl_coef, baseline = 0.15, 0.0
    RL_STEPS, RL_BS = 200, 64
    # Seed the curve with the SFT baseline (step 0) so the rise is visible.
    reward_curve = [{"step": 0, "avg_reward": sft_reward, "compliance": sft_comp}]

    for step in range(1, RL_STEPS + 1):
        prompts = [data.random_prompt(rng) for _ in range(RL_BS)]
        prefixes = torch.tensor([data.prefix_tokens(p) for p in prompts])
        # Sample responses from the current policy.
        torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=rng)))
        with torch.no_grad():
            gen = policy.generate(prefixes, data.RESP_LEN, temperature=1.0)
        responses = gen[:, prefixes.size(1):].tolist()
        full = data.encode_batch(prompts, responses)
        mb = response_mask.unsqueeze(0).expand(full.size(0), -1)

        logp = policy.sequence_logprob(full, mb)
        with torch.no_grad():
            logp_ref = sft_ref.sequence_logprob(full, mb)
            rm_score = rm(full)
            rm_norm = (rm_score - rm_score.mean()) / (rm_score.std() + 1e-6)
            kl = logp.detach() - logp_ref
            reward = rm_norm - kl_coef * kl
        baseline = 0.9 * baseline + 0.1 * reward.mean().item()
        adv = (reward - baseline).detach()
        loss = -(adv * logp).mean()
        opt_pi.zero_grad(); loss.backward(); opt_pi.step()

        if step % 20 == 0 or step == 1:
            avg_r, comp = eval_policy(policy, eval_prompts[:128], temperature=1.0)
            reward_curve.append({"step": step, "avg_reward": avg_r, "compliance": comp})
            print(f"  step {step:3d}/{RL_STEPS} | true avg reward {avg_r:.3f} | "
                  f"exact-compliance {comp*100:.1f}%")

    rlhf_reward, rlhf_comp = eval_policy(policy, eval_prompts, temperature=1.0)

    # best-of-n against the RM (rejection sampling), batched over prompts.
    bon_reward, bon_comp = best_of_n_eval(policy, rm, eval_prompts[:128], n=8)

    # ---------------------------------------------------------------- Results
    banner("RESULTS")
    print(f"  Reward-model preference accuracy      : {rm_acc*100:5.1f}%")
    print(f"  SFT   policy — avg reward (sampled)   : {sft_reward:.3f} | "
          f"compliance {sft_comp*100:.1f}%")
    print(f"  RLHF  policy — avg reward (sampled)   : {rlhf_reward:.3f} | "
          f"compliance {rlhf_comp*100:.1f}%")
    print(f"  RLHF + best-of-8 (RM rejection samp.) : {bon_reward:.3f} | "
          f"compliance {bon_comp*100:.1f}%")

    # One concrete example.
    ex_prompt = data.random_prompt(rng)
    prefix = torch.tensor([data.prefix_tokens(ex_prompt)])
    sft_out = sft_ref.generate(prefix, data.RESP_LEN, greedy=True)[0, prefix.size(1):].tolist()
    rlhf_out = policy.generate(prefix, data.RESP_LEN, greedy=True)[0, prefix.size(1):].tolist()
    print(f"\n  example prompt : {ex_prompt}")
    print(f"    ideal (rule) : {data.sorted_response(ex_prompt)}")
    print(f"    SFT  output  : {sft_out}")
    print(f"    RLHF output  : {rlhf_out}")

    # ---------------------------------------------------------------- Persist
    DATA_DIR.mkdir(exist_ok=True)
    payload = {
        "task": "sort tokens ascending (reward = sortedness)",
        "vocab_size": data.VOCAB_SIZE,
        "prompt_len": data.PROMPT_LEN,
        "rm_pref_accuracy": rm_acc,
        "sft": {"avg_reward": sft_reward, "compliance": sft_comp},
        "rlhf": {"avg_reward": rlhf_reward, "compliance": rlhf_comp},
        "best_of_n": {"n": 8, "avg_reward": bon_reward, "compliance": bon_comp},
        "reward_curve": reward_curve,
        "example": {
            "prompt": ex_prompt,
            "ideal": data.sorted_response(ex_prompt),
            "sft_output": sft_out,
            "rlhf_output": rlhf_out,
        },
    }
    out = DATA_DIR / "rlhf_results.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n  wrote {out}")
    print(f"  total time: {time.time() - t0:.1f}s")

    # ---------------------------------------------------------------- Asserts
    assert rm_acc > 0.9, f"RM accuracy too low: {rm_acc:.2f}"
    assert rlhf_reward > sft_reward + 0.02, (
        f"RLHF did not improve reward ({rlhf_reward:.3f} vs {sft_reward:.3f})"
    )
    assert bon_reward >= rlhf_reward - 1e-6
    print("\nOK: RM is accurate and RLHF increased rule-compliance over SFT.")


if __name__ == "__main__":
    main()
