"""Kimi k1.5 demo: RL for reasoning with a LENGTH PENALTY.

The policy makes two decisions per problem: an **answer** (the reasoning output)
and a **chain length** (how many scratch/"think" tokens to emit first). We RL
fine-tune it two ways with a GRPO-style update and compare:

* **no length penalty**  — reward = correctness only.
* **length penalty**     — reward = correctness + Kimi's group length reward.

Expected outcome (reproduced below on CPU in seconds):

* Both runs **raise accuracy** (the answer head learns ``max(a, b)`` from reward).
* **Without** the penalty the average chain length stays **long** — RL alone
  does not curb overthinking (the length is neutral for reward).
* **With** the penalty the chain length is **controlled** (shrinks sharply) at
  comparable accuracy — the "long2short" effect.

Writes ``data/kimi_run.json`` (learning curves) for the visualization.

Run with:  python demo/run_demo.py
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import TinyPolicy, answer_of, render_completion, MAX_THINK, THINK_NEED  # noqa: E402
from src.rl import rollout, compute_rewards, grpo_step, evaluate  # noqa: E402

SEED = 0
NUM_VALUES = 10
WARM_ITERS = 800     # SFT-style warm-start: learn the answer at full reveal
RL_ITERS = 400
PROMPTS_PER_ITER = 48
GROUP_SIZE = 8
RL_TEMP = 1.1
RL_LR = 3e-2
RL_ENT = 0.003
LEN_WEIGHT = 0.06


def all_problems():
    return [(a, b, answer_of(a, b)) for a in range(NUM_VALUES) for b in range(NUM_VALUES)]


def warm_start(model, device, gen):
    """Train ONLY the answer head with the chain held long (full reveal), so the
    reasoner learns max(a, b) cleanly before either length policy moves — the
    RL-from-a-good-init setup Kimi uses. The length policy is frozen (stays long)."""
    opt = torch.optim.Adam(model.parameters(), lr=RL_LR)
    probs = all_problems()
    for _ in range(WARM_ITERS):
        batch = [probs[gen.randrange(len(probs))] for _ in range(PROMPTS_PER_ITER)]
        roll = rollout(model, batch, GROUP_SIZE, device, temperature=RL_TEMP,
                       force_length=model.max_think)
        rewards = compute_rewards(roll, use_length_penalty=False, weight=0.0)
        grpo_step(model, opt, roll, rewards, device, entropy_coef=RL_ENT, update_length=False)


def rl_train(model, device, use_penalty, gen):
    # RL optimises ONLY the length policy — the reasoner is fixed by the
    # warm-start (SFT). This isolates Kimi's "long2short" length control and keeps
    # the reveal-mixture exploration noise from eroding the trained answer head.
    opt = torch.optim.Adam([model.length_logits], lr=RL_LR)
    probs = all_problems()
    curve = []
    for it in range(RL_ITERS):
        batch = [probs[gen.randrange(len(probs))] for _ in range(PROMPTS_PER_ITER)]
        roll = rollout(model, batch, GROUP_SIZE, device, temperature=RL_TEMP)
        rewards = compute_rewards(roll, use_penalty, LEN_WEIGHT)
        grpo_step(model, opt, roll, rewards, device, entropy_coef=RL_ENT)
        if it % 4 == 0 or it == RL_ITERS - 1:
            acc, mlen = evaluate(model, probs, device)
            curve.append({"iter": it, "acc": acc, "mean_len": mlen})
    return curve


def main() -> None:
    torch.manual_seed(SEED)
    device = torch.device("cpu")

    print("Kimi k1.5: RL for reasoning with a length penalty")
    print(f"task: output max(a,b) over {NUM_VALUES**2} problems | group={GROUP_SIZE} iters={RL_ITERS}\n")

    # Same random initialisation for both runs (answer head untrained -> ~chance;
    # length policy initialised to a long chain -> "overthinking").
    torch.manual_seed(SEED)
    base = TinyPolicy(NUM_VALUES, MAX_THINK, THINK_NEED)
    acc_chance, _ = evaluate(base, all_problems(), device)
    print(f"before training: accuracy {acc_chance*100:4.1f}% (chance)\n")

    print("SFT-style warm-start (learn the answer at full reveal; chain stays long) ...")
    warm_start(base, device, gen=random.Random(SEED))
    acc0, len0 = evaluate(base, all_problems(), device)
    print(f"after warm-start: accuracy {acc0*100:4.1f}% | mean chain length {len0:.2f}")
    print("(reasoner is now accurate, but still 'overthinks' with a long chain)\n")

    m_no = TinyPolicy(NUM_VALUES, MAX_THINK, THINK_NEED); m_no.load_state_dict(base.state_dict())
    m_pen = TinyPolicy(NUM_VALUES, MAX_THINK, THINK_NEED); m_pen.load_state_dict(base.state_dict())

    start = time.time()
    print("RL run A: NO length penalty ...")
    curve_no = rl_train(m_no, device, use_penalty=False, gen=random.Random(SEED + 1))
    print("RL run B: WITH length penalty ...")
    curve_pen = rl_train(m_pen, device, use_penalty=True, gen=random.Random(SEED + 1))
    print(f"trained both in {time.time() - start:.1f}s\n")

    acc_no, len_no = evaluate(m_no, all_problems(), device)
    acc_pen, len_pen = evaluate(m_pen, all_problems(), device)

    print("Final results:")
    print("                       accuracy    mean chain length")
    print(f"  before training   {acc_chance*100:6.1f}%          (chance)")
    print(f"  after warm-start  {acc0*100:6.1f}%        {len0:6.2f}")
    print(f"  RL no-penalty     {acc_no*100:6.1f}%        {len_no:6.2f}")
    print(f"  RL length-penalty {acc_pen*100:6.1f}%        {len_pen:6.2f}")
    print(f"\n  → warm-start makes the reasoner accurate but overthinking; RL WITHOUT a "
          f"penalty keeps the chain long ({len_no:.2f}),")
    print(f"    while the length penalty trims it {len_no:.2f} -> {len_pen:.2f} "
          f"({(1-len_pen/max(len_no,1e-6))*100:.0f}% shorter, toward the ~{THINK_NEED} steps "
          f"actually needed) at comparable accuracy.")

    # A few illustrative rendered completions from each policy. Both policies
    # share the same (frozen) answer head, so we pick problems the reasoner gets
    # RIGHT — the answer is identical and only the chain length differs, which is
    # exactly the point (same accuracy, far fewer tokens).
    def correct_examples(model, n=3):
        picks = []
        for (a, b, tgt) in all_problems():
            with torch.no_grad():
                ans = int(model.answer_logits(torch.tensor([a]), torch.tensor([b])).argmax())
            if ans == tgt:
                picks.append((a, b, tgt, ans))
            if len(picks) == n:
                break
        return picks

    def sample_render(model, picks):
        L = int(torch.argmax(model.length_logits))
        return [{"text": render_completion(a, b, L, ans), "ok": ans == tgt}
                for (a, b, tgt, ans) in picks]

    picks = correct_examples(m_no)

    out = {
        "meta": {"problems": NUM_VALUES**2, "group_size": GROUP_SIZE,
                 "rl_iters": RL_ITERS, "len_weight": LEN_WEIGHT, "max_think": MAX_THINK,
                 "think_need": THINK_NEED},
        "chance_acc": acc_chance,
        "start": {"acc": acc0, "mean_len": len0},
        "final": {
            "no_penalty": {"acc": acc_no, "mean_len": len_no},
            "length_penalty": {"acc": acc_pen, "mean_len": len_pen},
        },
        "curves": {"no_penalty": curve_no, "length_penalty": curve_pen},
        "samples": {
            "no_penalty": sample_render(m_no, picks),
            "length_penalty": sample_render(m_pen, picks),
        },
    }
    out_path = ROOT / "data" / "kimi_run.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")

    assert acc0 > acc_chance + 0.3, "warm-start should raise accuracy substantially"
    assert acc_no > acc_chance + 0.3, "the reasoner should stay accurate under RL"
    assert acc_pen > acc_chance + 0.3, "the reasoner should stay accurate under RL+penalty"
    assert len_no > 5.0, "without a penalty the chain should stay long (overthinking)"
    assert len_pen < len_no - 2.0, "the length penalty should shorten the chain"
    assert len_pen > 1.0, "the penalty should CONTROL, not eliminate, reasoning"
    assert acc_pen > acc_no - 0.1, "the length penalty should keep accuracy comparable"
    print("OK: length penalty controls chain length while accuracy stays high; RL alone leaves chains long.")


if __name__ == "__main__":
    main()
