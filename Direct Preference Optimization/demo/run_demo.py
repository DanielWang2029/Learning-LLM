"""Direct Preference Optimization end-to-end (Rafailov et al., 2023).

Aligns a tiny language model to a preference *directly*, with no reward model
and no RL sampling loop:

  1. SFT       — train a reference policy π_ref on (imperfect) demonstrations.
  2. DPO       — starting from a copy of π_ref, minimize the DPO loss on toy
                 preference pairs (chosen ≻ rejected by the sort rule).

Success criteria (asserted at the end):
  * the DPO loss decreases,
  * the implicit reward margin (chosen − rejected) increases and becomes
    positive on essentially all pairs, and
  * the policy's generations shift toward the preferred (sorted) behavior,
    beating the SFT reference — all without a separate reward model.

Runs on CPU in well under a minute.  Run with:  python demo/run_demo.py
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
torch.set_num_threads(1)

from src import data
from src.model import TinyLM
from src.dpo import dpo_loss

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SEED = 0
BETA = 0.1


def sft_loss(policy, full, mask):
    logits = policy(full[:, :-1])
    targets = full[:, 1:]
    m = mask[:, 1:]
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
    )
    return (loss * m.reshape(-1).float()).sum() / m.float().sum()


@torch.no_grad()
def eval_policy(policy, prompts, temperature=1.0, greedy=False):
    prefixes = torch.tensor([data.prefix_tokens(p) for p in prompts], dtype=torch.long)
    gen = policy.generate(prefixes, data.RESP_LEN, temperature=temperature, greedy=greedy)
    responses = gen[:, prefixes.size(1):].tolist()
    rewards = [data.sortedness(r) for r in responses]
    compliant = sum(data.is_compliant(p, r) for p, r in zip(prompts, responses))
    return sum(rewards) / len(rewards), compliant / len(prompts)


def banner(t):
    print("\n" + "=" * 66 + f"\n{t}\n" + "=" * 66)


def main() -> None:
    torch.manual_seed(SEED)
    rng = torch.Generator().manual_seed(SEED)
    t0 = time.time()

    print("Direct Preference Optimization in miniature — align a tiny LM by DPO")
    print(f"  vocab={data.VOCAB_SIZE}  prompt_len={data.PROMPT_LEN}  beta={BETA}")
    print("  preference rule: the sorted response is preferred (chosen ≻ rejected)")

    eval_prompts = [data.random_prompt(rng) for _ in range(256)]
    response_mask = data.build_response_mask()

    # ------------------------------------------------------------- SFT / ref
    banner("STAGE 1 — SFT reference policy π_ref (frozen)")
    ref = TinyLM(data.VOCAB_SIZE, data.SEQ_LEN)
    opt = torch.optim.Adam(ref.parameters(), lr=3e-3)
    p_dem, r_dem = data.make_demonstrations(2048, rng)
    full_dem = data.encode_batch(p_dem, r_dem)
    mask_b = response_mask.unsqueeze(0).expand(full_dem.size(0), -1)
    for step in range(1, 261):
        idx = torch.randint(0, full_dem.size(0), (64,), generator=rng)
        loss = sft_loss(ref, full_dem[idx], mask_b[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 65 == 0 or step == 1:
            print(f"  step {step:3d}/260 | sft loss {loss.item():.4f}")
    for prm in ref.parameters():
        prm.requires_grad_(False)
    ref.eval()

    ref_reward, ref_comp = eval_policy(ref, eval_prompts, temperature=1.0)
    print(f"\n  π_ref (sampled @T=1): avg reward {ref_reward:.3f} | "
          f"compliance {ref_comp*100:.1f}%")

    # ------------------------------------------------------------- DPO
    banner("STAGE 2 — Direct Preference Optimization (no reward model, no RL)")
    policy = copy.deepcopy(ref)
    for prm in policy.parameters():
        prm.requires_grad_(True)
    policy.train()
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)

    p_pref, chosen, rejected = data.make_preference_pairs(3072, rng)
    seq_ch = data.encode_batch(p_pref, chosen)
    seq_rj = data.encode_batch(p_pref, rejected)
    mask_ch = response_mask.unsqueeze(0).expand(seq_ch.size(0), -1)

    # Precompute frozen reference log-probs once (they never change).
    with torch.no_grad():
        ref_lp_ch_all = ref.sequence_logprob(seq_ch, mask_ch)
        ref_lp_rj_all = ref.sequence_logprob(seq_rj, mask_ch)

    DPO_STEPS, BS = 300, 128
    curve = [{"step": 0, "loss": None, "margin": 0.0, "r_chosen": 0.0,
              "r_rejected": 0.0, "acc": 0.5, "reward": ref_reward, "compliance": ref_comp}]
    for step in range(1, DPO_STEPS + 1):
        idx = torch.randint(0, seq_ch.size(0), (BS,), generator=rng)
        pol_ch = policy.sequence_logprob(seq_ch[idx], mask_ch[idx])
        pol_rj = policy.sequence_logprob(seq_rj[idx], mask_ch[idx])
        loss, r_ch, r_rj, margin = dpo_loss(
            pol_ch, pol_rj, ref_lp_ch_all[idx], ref_lp_rj_all[idx], BETA
        )
        opt.zero_grad(); loss.backward(); opt.step()

        if step % 25 == 0 or step == 1:
            acc = (margin > 0).float().mean().item()
            avg_r, comp = eval_policy(policy, eval_prompts[:128], temperature=1.0)
            curve.append({
                "step": step, "loss": loss.item(), "margin": margin.mean().item(),
                "r_chosen": r_ch.mean().item(), "r_rejected": r_rj.mean().item(),
                "acc": acc, "reward": avg_r, "compliance": comp,
            })
            print(f"  step {step:3d}/{DPO_STEPS} | dpo loss {loss.item():.4f} | "
                  f"margin {margin.mean().item():+.3f} | pref-acc {acc*100:5.1f}% | "
                  f"policy reward {avg_r:.3f}")

    policy.eval()
    dpo_reward, dpo_comp = eval_policy(policy, eval_prompts, temperature=1.0)

    # ------------------------------------------------------------- Results
    banner("RESULTS")
    print(f"  DPO loss                : {curve[1]['loss']:.4f} (start) -> "
          f"{curve[-1]['loss']:.4f} (end)")
    print(f"  implicit reward margin  : {curve[1]['margin']:+.3f} (start) -> "
          f"{curve[-1]['margin']:+.3f} (end)")
    print(f"  preference accuracy     : {curve[-1]['acc']*100:.1f}% of pairs have chosen ≻ rejected")
    print(f"  π_ref  policy — avg reward (sampled): {ref_reward:.3f} | "
          f"compliance {ref_comp*100:.1f}%")
    print(f"  π_DPO  policy — avg reward (sampled): {dpo_reward:.3f} | "
          f"compliance {dpo_comp*100:.1f}%")

    ex_prompt = data.random_prompt(rng)
    prefix = torch.tensor([data.prefix_tokens(ex_prompt)])
    ref_out = ref.generate(prefix, data.RESP_LEN, greedy=True)[0, prefix.size(1):].tolist()
    dpo_out = policy.generate(prefix, data.RESP_LEN, greedy=True)[0, prefix.size(1):].tolist()
    print(f"\n  example prompt : {ex_prompt}")
    print(f"    preferred    : {data.sorted_response(ex_prompt)}")
    print(f"    π_ref output : {ref_out}")
    print(f"    π_DPO output : {dpo_out}")

    # ------------------------------------------------------------- Persist
    DATA_DIR.mkdir(exist_ok=True)
    payload = {
        "task": "sort tokens ascending (chosen ≻ rejected)",
        "beta": BETA, "vocab_size": data.VOCAB_SIZE, "prompt_len": data.PROMPT_LEN,
        "ref": {"avg_reward": ref_reward, "compliance": ref_comp},
        "dpo": {"avg_reward": dpo_reward, "compliance": dpo_comp},
        "curve": curve,
        "example": {
            "prompt": ex_prompt, "preferred": data.sorted_response(ex_prompt),
            "ref_output": ref_out, "dpo_output": dpo_out,
        },
    }
    out = DATA_DIR / "dpo_results.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n  wrote {out}")
    print(f"  total time: {time.time()-t0:.1f}s")

    # ------------------------------------------------------------- Asserts
    assert curve[-1]["loss"] < curve[1]["loss"], "DPO loss did not decrease"
    assert curve[-1]["margin"] > curve[1]["margin"] + 0.5, "reward margin did not grow"
    assert curve[-1]["acc"] > 0.95, f"too few pairs satisfied chosen≻rejected: {curve[-1]['acc']:.2f}"
    assert dpo_reward > ref_reward + 0.02, (
        f"DPO did not shift generations toward preferred behavior "
        f"({dpo_reward:.3f} vs {ref_reward:.3f})"
    )
    print("\nOK: DPO loss fell, reward margin grew, and generations became more preferred.")


if __name__ == "__main__":
    main()
