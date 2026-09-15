"""DoRA end-to-end on CPU (Weight-Decomposed Low-Rank Adaptation, arXiv 2402.09353).

Story:
  1. PRETRAIN a small MLP on a base task and freeze it (the "pretrained model").
  2. ADAPT it to a related NEW task four ways and compare on a held-out test set:
        - Full fine-tuning : train every weight (the upper-bound reference).
        - LoRA             : W = W0 + (a/r)·B·A.
        - DoRA             : decompose W0 into magnitude m and direction V, then
                             train m directly and apply the low-rank update to V.
     At (nearly) matched trainable-parameter budgets, DoRA tracks full
     fine-tuning more closely than LoRA — the paper's headline claim.

Run with:  python demo/run_demo.py       (a few seconds on CPU)
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
torch.set_num_threads(1)  # shared box: avoid thread oversubscription

from src import data  # noqa: E402
from src.adapters import (  # noqa: E402
    inject_lora, inject_dora, adapter_parameters, DoRALinear,
)
from src.model import MLPClassifier, accuracy, count_trainable  # noqa: E402

DATA_DIR = ROOT / "data"
SEED = 0
RANK = 1
ALPHA = 2


def banner(t: str) -> None:
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def train(model, x, y, steps, lr, params=None, curve_x=None, curve=None,
          test_xy=None):
    params = list(params) if params is not None else list(model.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(SEED)
    model.train()
    for step in range(1, steps + 1):
        idx = torch.randint(0, x.size(0), (128,), generator=g)
        logits = model(x[idx])
        loss = loss_fn(logits, y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if curve is not None and (step % 40 == 0 or step == 1):
            curve_x.append(step)
            curve.append(round(accuracy(model, *test_xy), 4))
            model.train()
    return loss.item()


def main() -> None:
    torch.manual_seed(SEED)
    t0 = time.time()
    print("DoRA in miniature — magnitude/direction decomposition vs LoRA vs full-FT")
    print(f"  in_dim={data.IN_DIM} hidden={data.HIDDEN} classes={data.NUM_CLASSES}"
          f" | rank r={RANK} alpha={ALPHA}")

    xb_tr, yb_tr = data.make_task("base", 4000, seed=10)
    xb_te, yb_te = data.make_task("base", 2000, seed=11)
    xn_tr, yn_tr = data.make_task("new", 4000, seed=20)
    xn_te, yn_te = data.make_task("new", 2000, seed=21)
    test_new = (xn_te, yn_te)

    # ---- 1. Pretrain and freeze -------------------------------------------
    banner("STAGE 1 — Pretrain the base model, then FREEZE it (this is W0)")
    base = MLPClassifier(data.IN_DIM, data.HIDDEN, data.NUM_CLASSES)
    train(base, xb_tr, yb_tr, steps=2500, lr=3e-3)
    base_acc = accuracy(base, xb_te, yb_te)
    zero_shot_new = accuracy(base, xn_te, yn_te)
    print(f"  base-task accuracy of pretrained model : {base_acc*100:.1f}%")
    print(f"  new-task accuracy BEFORE adaptation     : {zero_shot_new*100:.1f}%")

    ADAPT_STEPS, ADAPT_LR = 1500, 5e-3

    # ---- 2a. Full fine-tuning (upper bound) --------------------------------
    banner("STAGE 2a — Full fine-tuning (train ALL weights) — upper bound")
    full = copy.deepcopy(base)
    for p in full.parameters():
        p.requires_grad_(True)
    full_params = count_trainable(full)
    fc_x, fc = [], []
    train(full, xn_tr, yn_tr, ADAPT_STEPS, 3e-3, curve_x=fc_x, curve=fc,
          test_xy=test_new)
    full_acc = accuracy(full, xn_te, yn_te)
    print(f"  trainable params : {full_params:,}")
    print(f"  new-task accuracy: {full_acc*100:.1f}%")

    # ---- 2b. LoRA ----------------------------------------------------------
    banner("STAGE 2b — LoRA (W = W0 + (a/r)·B·A)")
    lora = copy.deepcopy(base)
    for p in lora.parameters():
        p.requires_grad_(False)
    wrapped = inject_lora(lora, r=RANK, alpha=ALPHA)
    lora_params = count_trainable(lora)
    lc_x, lc = [], []
    train(lora, xn_tr, yn_tr, ADAPT_STEPS, ADAPT_LR,
          params=adapter_parameters(lora), curve_x=lc_x, curve=lc, test_xy=test_new)
    lora_acc = accuracy(lora, xn_te, yn_te)
    print(f"  wrapped layers   : {wrapped}")
    print(f"  trainable params : {lora_params:,}")
    print(f"  new-task accuracy: {lora_acc*100:.1f}%")

    # ---- 2c. DoRA ----------------------------------------------------------
    banner("STAGE 2c — DoRA (train magnitude m + low-rank direction update)")
    dora = copy.deepcopy(base)
    for p in dora.parameters():
        p.requires_grad_(False)
    inject_dora(dora, r=RANK, alpha=ALPHA)
    dora_params = count_trainable(dora)
    dc_x, dc = [], []
    train(dora, xn_tr, yn_tr, ADAPT_STEPS, ADAPT_LR,
          params=adapter_parameters(dora), curve_x=dc_x, curve=dc, test_xy=test_new)
    dora_acc = accuracy(dora, xn_te, yn_te)
    print(f"  trainable params : {dora_params:,}  "
          f"(+{dora_params-lora_params:,} magnitude scalars vs LoRA)")
    print(f"  new-task accuracy: {dora_acc*100:.1f}%")

    # ---- Results -----------------------------------------------------------
    banner("RESULTS — new-task accuracy (held-out)")
    print(f"  full fine-tuning : {full_params:>7,} params -> {full_acc*100:5.1f}%  (upper bound)")
    print(f"  DoRA (r={RANK})       : {dora_params:>7,} params -> {dora_acc*100:5.1f}%")
    print(f"  LoRA (r={RANK})       : {lora_params:>7,} params -> {lora_acc*100:5.1f}%")
    print(f"  gap to full-FT   : DoRA {(full_acc-dora_acc)*100:+.1f}  |  "
          f"LoRA {(full_acc-lora_acc)*100:+.1f}")
    print(f"  DoRA beats LoRA by {(dora_acc-lora_acc)*100:+.1f} points at ~matched budget")

    # ---- Decomposition illustration for one layer --------------------------
    # Show how DoRA moved magnitude vs direction on fc1 (relative to W0).
    layer = dora.fc1
    assert isinstance(layer, DoRALinear)
    with torch.no_grad():
        W0 = layer.weight
        m0 = W0.norm(dim=1)
        m_final = layer.magnitude
        dv = layer.scaling * (layer.lora_B @ layer.lora_A)
        dir_change = dv.norm(dim=1) / W0.norm(dim=1)   # relative directional move
        mag_change = (m_final - m0) / m0               # relative magnitude move
    decomp = {
        "magnitude_change": [round(v, 4) for v in mag_change[:24].tolist()],
        "direction_change": [round(v, 4) for v in dir_change[:24].tolist()],
    }

    # ---- Persist -----------------------------------------------------------
    DATA_DIR.mkdir(exist_ok=True)
    payload = {
        "config": {
            "in_dim": data.IN_DIM, "hidden": data.HIDDEN,
            "classes": data.NUM_CLASSES, "rank": RANK, "alpha": ALPHA,
        },
        "base_task_accuracy": base_acc,
        "new_task_zero_shot": zero_shot_new,
        "methods": {
            "full":  {"params": full_params, "accuracy": full_acc},
            "dora":  {"params": dora_params, "accuracy": dora_acc},
            "lora":  {"params": lora_params, "accuracy": lora_acc},
        },
        "wrapped_layers": wrapped,
        "curves": {
            "steps_full": fc_x, "full": fc,
            "steps_lora": lc_x, "lora": lc,
            "steps_dora": dc_x, "dora": dc,
        },
        "decomposition_fc1": decomp,
    }
    out = DATA_DIR / "dora_results.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n  wrote {out}")
    print(f"  total time: {time.time()-t0:.1f}s")

    # ---- Asserts -----------------------------------------------------------
    assert base_acc > 0.82, f"base model did not learn base task ({base_acc:.2f})"
    assert full_acc > 0.82, f"full-FT should learn the new task ({full_acc:.2f})"
    assert dora_acc > lora_acc, (
        f"DoRA ({dora_acc:.3f}) should beat LoRA ({lora_acc:.3f}) at matched budget")
    assert abs(dora_params - lora_params) <= data.HIDDEN * 2 + data.NUM_CLASSES, (
        "DoRA should only add the magnitude scalars over LoRA")
    print("\nOK: DoRA tracks full fine-tuning more closely than LoRA at a "
          "matched trainable-parameter budget.")


if __name__ == "__main__":
    main()
