"""A Survey of Large Language Models — the LLM lifecycle in miniature, on CPU.

Zhao et al. (2023) survey the field, not a single algorithm. Its organizing
spine is the *lifecycle* of a modern LLM. We run that entire spine end-to-end on
one tiny GPT and print a metric after each stage, so the progression is visible:

  Stage 1  Pre-training      -> language-modeling loss drops
  Stage 2  Adaptation (SFT)  -> instruction exact-match jumps up
  Stage 3  Alignment (DPO)   -> preference for the "chosen" answer increases

Runs on CPU in a few seconds. Writes data/lifecycle.json for the visualization
(which also hosts the survey's technique taxonomy).

Run with:  python demo/run_demo.py
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(1)  # shared box: avoid CPU oversubscription

PAPER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PAPER_DIR))

from src.model import GPT, GPTConfig  # noqa: E402
from src.pipeline import (BLOCK, VOCAB_SIZE, dpo_loss, exact_match,  # noqa: E402
                          lm_loss, max_eval_batch, preference_accuracy,
                          preference_batch, pretrain_batch, seq_logprob,
                          sft_batch, train_lm)
import torch as _torch  # noqa: E402

DATA_DIR = PAPER_DIR / "data"


def main() -> None:
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=VOCAB_SIZE, block_size=BLOCK,
                          n_layer=2, n_head=4, n_embd=48))

    print("=" * 70)
    print("A Survey of Large Language Models — the LLM lifecycle in miniature")
    print("=" * 70)
    print("Survey, not an algorithm: we run every stage the survey covers on ONE")
    print("tiny GPT over a shared 13-token vocabulary.\n")

    # Held-out evaluation sets (fixed across stages).
    sft_eval = sft_batch(300, seed=90_000)          # SORT instruction
    pref_eval = preference_batch(300, seed=91_000)  # MAX preference pairs
    max_eval = max_eval_batch(300, seed=92_000)     # MAX exact-match

    t0 = time.time()

    # --- Stage 1: PRE-TRAINING ------------------------------------------
    pre_first, pre_last = train_lm(model, lambda s: pretrain_batch(256, s),
                                   steps=250, lr=3e-3, seed=0)
    acc_after_pretrain = exact_match(model, *sft_eval)
    print("[Stage 1] Pre-training (next-token prediction on a raw corpus)")
    print(f"          LM loss: {pre_first:.3f} -> {pre_last:.3f}")
    print(f"          instruction exact-match (untuned): "
          f"{acc_after_pretrain*100:.1f}%\n")

    # --- Stage 2: SUPERVISED FINE-TUNING --------------------------------
    _, sft_last = train_lm(model, lambda s: sft_batch(256, s),
                           steps=300, lr=2e-3, seed=10_000)
    acc_after_sft = exact_match(model, *sft_eval)
    pref_after_sft = preference_accuracy(model, *pref_eval)
    print("[Stage 2] Adaptation / SFT (instruction -> answer: 'sort 4 digits')")
    print(f"          SFT loss (final): {sft_last:.3f}")
    print(f"          SORT instruction exact-match: {acc_after_pretrain*100:.1f}% "
          f"-> {acc_after_sft*100:.1f}%")
    print(f"          MAX preference accuracy (never SFT'd, pre-DPO): "
          f"{pref_after_sft*100:.1f}%\n")

    # --- Stage 3: ALIGNMENT via DPO -------------------------------------
    ref = copy.deepcopy(model)                 # frozen reference = SFT model
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
    model.train()
    beta, dpo_steps, replay_w = 0.3, 250, 1.0
    for step in range(dpo_steps):
        cs, cm, rs, rm = preference_batch(128, seed=20_000 + step)
        loss = dpo_loss(model, ref, cs, cm, rs, rm, beta)
        # SFT replay: keep the SORT skill while aligning (cf. InstructGPT PPO-ptx).
        rx, rmask = sft_batch(128, seed=30_000 + step)
        loss = loss + replay_w * lm_loss(
            model, _torch.tensor(rx), _torch.tensor(rmask)
        )
        opt.zero_grad(); loss.backward(); opt.step()
    pref_after_dpo = preference_accuracy(model, *pref_eval)
    max_acc_after_dpo = exact_match(model, *max_eval)
    sort_acc_after_dpo = exact_match(model, *sft_eval)
    # reward margin = mean (logp_chosen - logp_rejected) on held-out pairs
    with torch.no_grad():
        margin = (seq_logprob(model, pref_eval[0], pref_eval[1])
                  - seq_logprob(model, pref_eval[2], pref_eval[3])).mean().item()
    print("[Stage 3] Alignment / DPO (learn the MAX behavior from preferences only)")
    print(f"          MAX preference accuracy: {pref_after_sft*100:.1f}% "
          f"-> {pref_after_dpo*100:.1f}%")
    print(f"          held-out reward margin (logp_chosen - logp_rejected): "
          f"{margin:+.2f}")
    print(f"          MAX exact-match, learned from comparisons: "
          f"{max_acc_after_dpo*100:.1f}%")
    print(f"          SORT skill retained from SFT: {sort_acc_after_dpo*100:.1f}%")

    elapsed = time.time() - t0
    print(f"\nFull lifecycle completed in {elapsed:.1f}s")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "stages": [
            {"stage": "Pre-training", "metric": "LM loss (final)",
             "value": pre_last, "detail": f"{pre_first:.3f} -> {pre_last:.3f}",
             "kind": "loss"},
            {"stage": "SFT", "metric": "SORT exact-match",
             "value": acc_after_sft,
             "detail": f"{acc_after_pretrain*100:.1f}% -> {acc_after_sft*100:.1f}%",
             "kind": "acc"},
            {"stage": "DPO alignment", "metric": "MAX preference accuracy",
             "value": pref_after_dpo,
             "detail": f"{pref_after_sft*100:.1f}% -> {pref_after_dpo*100:.1f}%",
             "kind": "acc"},
        ],
        "sort_accuracy": {"pretrained": acc_after_pretrain, "sft": acc_after_sft,
                          "dpo": sort_acc_after_dpo},
        "max_accuracy": {"dpo": max_acc_after_dpo},
        "preference_accuracy": {"sft": pref_after_sft, "dpo": pref_after_dpo},
        "reward_margin": margin,
        "pretrain_loss": {"first": pre_first, "last": pre_last},
    }
    (DATA_DIR / "lifecycle.json").write_text(json.dumps(out, indent=2))
    print(f"Wrote {DATA_DIR / 'lifecycle.json'}")

    ok = (acc_after_sft > acc_after_pretrain + 0.3
          and pref_after_dpo > pref_after_sft + 0.1)
    if not ok:
        raise SystemExit("Lifecycle did not progress as expected.")
    print("OK: one model walked the full pre-train -> SFT -> align lifecycle.")


if __name__ == "__main__":
    main()
