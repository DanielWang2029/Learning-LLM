"""LoRA end-to-end on CPU (Hu et al., 2021).

Story, in three acts:

  1. PRETRAIN a small MLP on a *base* task and freeze it (this is our
     "pretrained model" W0).
  2. ADAPT it to a *new* task two ways and compare:
        - Full fine-tuning : train every weight.
        - LoRA             : freeze W0, train only the rank-r factors B,A.
     LoRA trains ~100x fewer parameters yet reaches accuracy close to full
     fine-tuning (paper's central claim, Section 1 & 5).
  3. MERGE / UNMERGE / SWAP the adapter to show ΔW=BA folds into W0 with no
     inference cost and can be toggled on and off (Section 3, "no additional
     inference latency").

Runs on CPU in a few seconds.  Run with:  python demo/run_demo.py
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
torch.set_num_threads(1)  # shared box: avoid thread oversubscription

from src import data
from src.lora import inject_lora, lora_parameters, merge_all, unmerge_all
from src.model import MLPClassifier, accuracy, count_total, count_trainable

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SEED = 0
RANK = 2
ALPHA = 8


def banner(t: str) -> None:
    print("\n" + "=" * 68 + f"\n{t}\n" + "=" * 68)


def train(model, x, y, steps, lr, params=None):
    params = list(params) if params is not None else list(model.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(SEED)
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, x.size(0), (128,), generator=g)
        logits = model(x[idx])
        loss = loss_fn(logits, y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item()


def main() -> None:
    torch.manual_seed(SEED)
    t0 = time.time()

    print("LoRA in miniature — adapt a frozen pretrained MLP to a new task")
    print(f"  in_dim={data.IN_DIM} hidden={data.HIDDEN} classes={data.NUM_CLASSES}"
          f" | LoRA rank r={RANK} alpha={ALPHA}")

    # Data: base task (for pretraining) and a related new task (for adaptation).
    xb_tr, yb_tr = data.make_task("base", 4000, seed=10)
    xb_te, yb_te = data.make_task("base", 2000, seed=11)
    xn_tr, yn_tr = data.make_task("new", 4000, seed=20)
    xn_te, yn_te = data.make_task("new", 2000, seed=21)

    # ----------------------------------------------------------- 1. Pretrain
    banner("STAGE 1 — Pretrain the base model, then FREEZE it (this is W0)")
    base = MLPClassifier(data.IN_DIM, data.HIDDEN, data.NUM_CLASSES)
    train(base, xb_tr, yb_tr, steps=1200, lr=3e-3)
    base_acc = accuracy(base, xb_te, yb_te)
    print(f"  base-task accuracy of pretrained model : {base_acc*100:.1f}%")
    # Its accuracy on the NEW task before any adaptation (the starting point).
    zero_shot_new = accuracy(base, xn_te, yn_te)
    print(f"  new-task accuracy BEFORE adaptation     : {zero_shot_new*100:.1f}%")

    # ----------------------------------------------------------- 2a. Full FT
    banner("STAGE 2a — Full fine-tuning on the new task (train ALL weights)")
    full = copy.deepcopy(base)
    for p in full.parameters():
        p.requires_grad_(True)
    full_trainable = count_trainable(full)
    train(full, xn_tr, yn_tr, steps=1000, lr=3e-3)
    full_acc = accuracy(full, xn_te, yn_te)
    print(f"  trainable params : {full_trainable:,}")
    print(f"  new-task accuracy: {full_acc*100:.1f}%")

    # ----------------------------------------------------------- 2b. LoRA
    banner("STAGE 2b — LoRA fine-tuning (freeze W0, train only B and A)")
    lora_model = copy.deepcopy(base)
    for p in lora_model.parameters():
        p.requires_grad_(False)          # freeze the pretrained weights
    wrapped = inject_lora(lora_model, r=RANK, alpha=ALPHA)
    lora_trainable = count_trainable(lora_model)
    print(f"  wrapped layers   : {wrapped}")
    train(lora_model, xn_tr, yn_tr, steps=1200, lr=1e-2,
          params=lora_parameters(lora_model))
    lora_acc = accuracy(lora_model, xn_te, yn_te)
    ratio = full_trainable / lora_trainable
    print(f"  trainable params : {lora_trainable:,}  "
          f"({ratio:.1f}x fewer than full fine-tuning)")
    print(f"  new-task accuracy: {lora_acc*100:.1f}%")

    # ----------------------------------------------------------- 3. Merge/swap
    banner("STAGE 3 — Merge / unmerge / swap the adapter (no inference cost)")
    acc_unmerged = accuracy(lora_model, xn_te, yn_te)
    merge_all(lora_model)                # fold ΔW = (alpha/r)·B·A into W0
    acc_merged = accuracy(lora_model, xn_te, yn_te)
    unmerge_all(lora_model)             # restore W0; ΔW removed
    acc_after_unmerge = accuracy(lora_model, xn_te, yn_te)
    acc_base_restored = None
    # With the adapter unmerged AND its factors zeroed we recover the base model.
    swapped = copy.deepcopy(lora_model)
    for m in swapped.modules():
        if hasattr(m, "lora_B"):
            with torch.no_grad():
                m.lora_B.zero_()
    acc_base_restored = accuracy(swapped, xn_te, yn_te)
    print(f"  accuracy (LoRA active, unmerged) : {acc_unmerged*100:.1f}%")
    print(f"  accuracy (ΔW merged into W0)     : {acc_merged*100:.1f}%   "
          f"-> identical: {abs(acc_unmerged-acc_merged) < 1e-6}")
    print(f"  accuracy (adapter unmerged again): {acc_after_unmerge*100:.1f}%")
    print(f"  accuracy (adapter swapped OFF)   : {acc_base_restored*100:.1f}%   "
          f"(back to the un-adapted model)")

    # ----------------------------------------------------------- Results
    banner("RESULTS")
    print(f"  full fine-tuning : {full_trainable:,} params -> {full_acc*100:.1f}%")
    print(f"  LoRA (r={RANK})      : {lora_trainable:,} params -> {lora_acc*100:.1f}%")
    print(f"  parameter saving : {ratio:.1f}x fewer trainable params")
    print(f"  accuracy gap     : {(full_acc-lora_acc)*100:+.1f} points")

    # ----------------------------------------------------------- Persist
    DATA_DIR.mkdir(exist_ok=True)
    delta_norm = 0.0
    with torch.no_grad():
        for m in lora_model.modules():
            if hasattr(m, "delta_w"):
                delta_norm += float(m.delta_w().norm())
    payload = {
        "config": {
            "in_dim": data.IN_DIM, "hidden": data.HIDDEN,
            "classes": data.NUM_CLASSES, "rank": RANK, "alpha": ALPHA,
        },
        "base_task_accuracy": base_acc,
        "new_task_zero_shot": zero_shot_new,
        "full": {"trainable": full_trainable, "accuracy": full_acc},
        "lora": {"trainable": lora_trainable, "accuracy": lora_acc,
                 "delta_w_norm": delta_norm},
        "param_ratio": ratio,
        "wrapped_layers": wrapped,
        "merge_check": {
            "unmerged": acc_unmerged, "merged": acc_merged,
            "unmerged_again": acc_after_unmerge, "adapter_off": acc_base_restored,
        },
    }
    out = DATA_DIR / "lora_results.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n  wrote {out}")
    print(f"  total time: {time.time()-t0:.1f}s")

    # ----------------------------------------------------------- Asserts
    assert base_acc > 0.85, f"base model did not learn base task ({base_acc:.2f})"
    assert ratio > 20, f"LoRA should train far fewer params (got {ratio:.1f}x)"
    assert lora_acc > 0.80, f"LoRA did not learn the new task ({lora_acc:.2f})"
    assert lora_acc >= full_acc - 0.07, (
        f"LoRA accuracy {lora_acc:.2f} not close to full FT {full_acc:.2f}")
    assert abs(acc_unmerged - acc_merged) < 1e-6, "merge changed the output"
    assert abs(acc_after_unmerge - acc_unmerged) < 1e-6, "unmerge not exact"
    print("\nOK: LoRA matched full fine-tuning with ~{:.0f}x fewer trainable "
          "params, and merge/unmerge is exact.".format(ratio))


if __name__ == "__main__":
    main()
