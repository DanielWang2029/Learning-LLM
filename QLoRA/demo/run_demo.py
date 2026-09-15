"""QLoRA end-to-end on CPU (Dettmers et al., 2023).

Builds directly on LoRA. Story, in four acts:

  1. PRETRAIN a small MLP on a base task (this is the "large" model, in fp32).
  2. QUANTIZE its frozen weights to 4-bit NF4 and measure the quantization
     error and the accuracy retained by the 4-bit model.
  3. FINE-TUNE with LoRA *on top of the quantized frozen base* on a new task,
     and show it still learns to high accuracy (adapters stay in fp32).
  4. MEMORY: compare bytes for fp32 / fp16 base vs 4-bit base + fp32 adapters.

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
from src.model import MLPClassifier, accuracy, count_trainable, linear_weight_numel
from src.qlora import inject_qlora, qlora_parameters
from src.quant import NF4_CODEBOOK, bytes_4bit, quantization_error

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SEED = 0
RANK = 2
ALPHA = 8
BLOCK = 64


def banner(t: str) -> None:
    print("\n" + "=" * 68 + f"\n{t}\n" + "=" * 68)


def train(model, x, y, steps, lr, params):
    opt = torch.optim.Adam(list(params), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(SEED)
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, x.size(0), (128,), generator=g)
        loss = loss_fn(model(x[idx]), y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()


def human(nbytes: float) -> str:
    return f"{nbytes/1024:.1f} KB" if nbytes < 1024 * 1024 else f"{nbytes/1024/1024:.2f} MB"


def main() -> None:
    torch.manual_seed(SEED)
    t0 = time.time()

    print("QLoRA in miniature — 4-bit NF4 frozen base + fp32 LoRA adapters")
    print(f"  in_dim={data.IN_DIM} hidden={data.HIDDEN} classes={data.NUM_CLASSES}"
          f" | rank r={RANK} alpha={ALPHA} | NF4 block={BLOCK}")

    xb_tr, yb_tr = data.make_task("base", 4000, seed=10)
    xb_te, yb_te = data.make_task("base", 2000, seed=11)
    xn_tr, yn_tr = data.make_task("new", 4000, seed=20)
    xn_te, yn_te = data.make_task("new", 2000, seed=21)

    # ----------------------------------------------------------- 1. Pretrain
    banner("STAGE 1 — Pretrain the fp32 base model, then FREEZE it")
    base = MLPClassifier(data.IN_DIM, data.HIDDEN, data.NUM_CLASSES)
    train(base, xb_tr, yb_tr, steps=1200, lr=3e-3, params=base.parameters())
    fp32_base_acc = accuracy(base, xb_te, yb_te)
    print(f"  fp32 base-task accuracy : {fp32_base_acc*100:.1f}%")

    # ----------------------------------------------------------- 2. Quantize
    banner("STAGE 2 — Quantize the frozen base weights to 4-bit NF4")
    per_layer = {}
    for name, m in base.named_modules():
        if isinstance(m, nn.Linear):
            err = quantization_error(m.weight.data, BLOCK)
            per_layer[name] = err
            print(f"  {name:5s} weight {tuple(m.weight.shape)!s:12s} "
                  f"quant error (rel L2): {err*100:.2f}%")

    # Build the quantized model and measure accuracy retained by 4-bit weights.
    qbase = copy.deepcopy(base)
    inject_qlora(qbase, r=RANK, alpha=ALPHA, block_size=BLOCK)  # adapters = 0 -> no-op
    quant_base_acc = accuracy(qbase, xb_te, yb_te)
    overall_err = quantization_error(
        torch.cat([m.weight.data.reshape(-1) for m in base.modules()
                   if isinstance(m, nn.Linear)]), BLOCK)
    print(f"\n  overall weight quant error (rel L2): {overall_err*100:.2f}%")
    print(f"  base-task accuracy: fp32 {fp32_base_acc*100:.1f}%  ->  "
          f"4-bit {quant_base_acc*100:.1f}%  "
          f"(drop {abs(fp32_base_acc-quant_base_acc)*100:.1f} pts)")

    # ----------------------------------------------------------- 3. Fine-tune
    banner("STAGE 3 — Fine-tune LoRA on the 4-bit frozen base (new task)")
    zero_shot_new = accuracy(qbase, xn_te, yn_te)
    print(f"  new-task accuracy BEFORE adaptation : {zero_shot_new*100:.1f}%")
    qlora_trainable = count_trainable(qbase)
    train(qbase, xn_tr, yn_tr, steps=1200, lr=1e-2, params=qlora_parameters(qbase))
    qlora_acc = accuracy(qbase, xn_te, yn_te)
    print(f"  trainable (fp32 LoRA) params : {qlora_trainable:,}")
    print(f"  new-task accuracy AFTER QLoRA: {qlora_acc*100:.1f}%")

    # ----------------------------------------------------------- 4. Memory
    banner("STAGE 4 — Memory: fp32 / fp16 base vs 4-bit base + fp32 adapters")
    base_numel = linear_weight_numel(base)
    fp32_bytes = base_numel * 4
    fp16_bytes = base_numel * 2
    nf4_bytes = bytes_4bit(base_numel, BLOCK)
    adapter_bytes = qlora_trainable * 4          # LoRA factors kept in fp32
    qlora_total = nf4_bytes + adapter_bytes
    eff_bits = nf4_bytes / base_numel * 8
    print(f"  base Linear weights: {base_numel:,} values")
    print(f"  fp32 base                     : {human(fp32_bytes)}")
    print(f"  fp16 base                     : {human(fp16_bytes)}")
    print(f"  4-bit NF4 base (+absmax)      : {human(nf4_bytes)}  "
          f"(~{eff_bits:.2f} bits/value)")
    print(f"  + fp32 LoRA adapters          : {human(adapter_bytes)}")
    print(f"  = QLoRA total (base+adapters) : {human(qlora_total)}")
    print(f"  memory vs fp32 base           : {fp32_bytes/qlora_total:.1f}x smaller")
    print(f"  memory vs fp16 base           : {fp16_bytes/qlora_total:.1f}x smaller")

    # ----------------------------------------------------------- Results
    banner("RESULTS")
    print(f"  quantization error (overall)  : {overall_err*100:.2f}% (rel L2)")
    print(f"  4-bit base keeps base-task acc : {quant_base_acc*100:.1f}% "
          f"(fp32 was {fp32_base_acc*100:.1f}%)")
    print(f"  QLoRA new-task accuracy        : {qlora_acc*100:.1f}% "
          f"(from {zero_shot_new*100:.1f}% before)")
    print(f"  memory saving vs fp32          : {fp32_bytes/qlora_total:.1f}x")

    # ----------------------------------------------------------- Persist
    DATA_DIR.mkdir(exist_ok=True)
    payload = {
        "config": {"in_dim": data.IN_DIM, "hidden": data.HIDDEN,
                   "classes": data.NUM_CLASSES, "rank": RANK, "alpha": ALPHA,
                   "block_size": BLOCK},
        "quant_error_overall": overall_err,
        "quant_error_per_layer": per_layer,
        "fp32_base_acc": fp32_base_acc,
        "quant_base_acc": quant_base_acc,
        "new_task_zero_shot": zero_shot_new,
        "qlora_acc": qlora_acc,
        "qlora_trainable": qlora_trainable,
        "memory": {
            "base_numel": base_numel,
            "fp32_bytes": fp32_bytes, "fp16_bytes": fp16_bytes,
            "nf4_bytes": nf4_bytes, "adapter_bytes": adapter_bytes,
            "qlora_total": qlora_total, "eff_bits": eff_bits,
            "save_vs_fp32": fp32_bytes / qlora_total,
            "save_vs_fp16": fp16_bytes / qlora_total,
        },
        "codebook": [round(float(v), 6) for v in NF4_CODEBOOK.tolist()],
    }
    out = DATA_DIR / "qlora_results.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n  wrote {out}")
    print(f"  total time: {time.time()-t0:.1f}s")

    # ----------------------------------------------------------- Asserts
    assert fp32_base_acc > 0.85, f"base did not learn base task ({fp32_base_acc:.2f})"
    assert overall_err < 0.15, f"4-bit quant error too high ({overall_err:.3f})"
    assert quant_base_acc > fp32_base_acc - 0.05, "4-bit base lost too much accuracy"
    assert qlora_acc > 0.78, f"QLoRA did not learn the new task ({qlora_acc:.2f})"
    assert fp32_bytes / qlora_total > 4, "expected >4x memory saving over fp32"
    print("\nOK: 4-bit base kept its accuracy, LoRA-on-quantized-base learned the "
          f"new task ({qlora_acc*100:.1f}%), at {fp32_bytes/qlora_total:.1f}x less memory.")


if __name__ == "__main__":
    main()
