"""End-to-end hybrid Mamba-Attention demo for Nemotron 3 Super (CPU, < 60s).

We train a small **hybrid** stack (mostly linear Mamba-SSM layers with one
attention layer) on a long-range recall task, showing it learns to carry
information across the whole sequence. We then time an SSM layer against an
attention layer at growing sequence lengths to make the **linear vs quadratic**
cost gap concrete.

Run with:  python demo/run_demo.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))

from src import CausalAttention, HybridModel, SelectiveSSM  # noqa: E402
import generate_data as gd  # noqa: E402


def evaluate(model, seq_len, rng, trials=400):
    model.eval()
    with torch.no_grad():
        toks, tgt = gd.make_batch(trials, seq_len, rng)
        pred = model(toks)[:, -1].argmax(dim=-1)
        return (pred == tgt).float().mean().item()


def time_layer(layer, L, d_model, batch=8, reps=3):
    x = torch.randn(batch, L, d_model)
    with torch.no_grad():
        layer(x)  # warm-up
        t0 = time.time()
        for _ in range(reps):
            layer(x)
        return (time.time() - t0) / reps


def main() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    t0 = time.time()

    seq_len = 48
    d_model = 64
    model = HybridModel(gd.VOCAB, d_model=d_model, pattern="MMA", repeats=1,
                        n_heads=4, d_state=8)
    n_params = sum(p.numel() for p in model.parameters())
    n_ssm = model.layer_kinds.count("M")
    n_attn = model.layer_kinds.count("A")
    print(f"Hybrid stack: {model.describe()}")
    print(f"{n_params:,} params | {n_ssm} SSM (linear) + {n_attn} attention (quadratic) layers")
    print(f"task: recall a token from position 0 across a length-{seq_len} sequence\n")

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    acc_curve = []
    for step in range(1, 401):
        model.train()
        toks, tgt = gd.make_batch(64, seq_len, rng)
        logits = model(toks)[:, -1]
        loss = torch.nn.functional.cross_entropy(logits, tgt)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == 1:
            acc = evaluate(model, seq_len, np.random.default_rng(123))
            acc_curve.append({"step": step, "acc": acc})
            print(f"step {step:3d} | loss {loss.item():.4f} | recall acc {acc*100:5.1f}%")

    final_acc = evaluate(model, seq_len, np.random.default_rng(999))
    chance = 1.0 / gd.N_CONTENT

    # --- Cost contrast: SSM (linear) vs attention (quadratic) forward time. ---
    print("\nForward time: one SSM layer vs one attention layer")
    ssm = SelectiveSSM(d_model, d_state=8)
    attn = CausalAttention(d_model, n_heads=4)
    lengths = [32, 64, 128, 256]
    ssm_ms, attn_ms = [], []
    print(f"{'L':>6} | {'SSM (ms)':>10} | {'Attn (ms)':>10} | {'ratio':>7}")
    for L in lengths:
        s = time_layer(ssm, L, d_model)
        a = time_layer(attn, L, d_model)
        ssm_ms.append(s * 1000)
        attn_ms.append(a * 1000)
        print(f"{L:6d} | {s*1000:10.2f} | {a*1000:10.2f} | {a/s:6.2f}x")

    print("\n=== Results ===")
    print(f"final recall accuracy : {final_acc*100:.1f}%  (chance = {chance*100:.1f}%)")
    print(f"attention/SSM time ratio grows with L: "
          f"{attn_ms[0]/ssm_ms[0]:.2f}x (L=32) -> {attn_ms[-1]/ssm_ms[-1]:.2f}x (L={lengths[-1]})")
    print(f"elapsed: {time.time()-t0:.1f}s")

    curve_L = [64, 128, 256, 512, 1024, 2048, 4096]
    out = {
        "seq_len": seq_len,
        "pattern": model.describe(),
        "layer_kinds": model.layer_kinds,
        "n_ssm": n_ssm,
        "n_attn": n_attn,
        "final_acc": final_acc,
        "chance": chance,
        "acc_curve": acc_curve,
        "timing": {"lengths": lengths, "ssm_ms": [round(x, 3) for x in ssm_ms],
                   "attn_ms": [round(x, 3) for x in attn_ms]},
        "cost_curve": {"lengths": curve_L,
                       "ssm_linear": [n for n in curve_L],
                       "attn_quadratic": [n * n // curve_L[0] for n in curve_L]},
        "vocab": gd.VOCAB,
    }
    art = ROOT / "data" / "hybrid_result.json"
    art.write_text(json.dumps(out))
    print(f"\nWrote {art}")

    assert final_acc > 0.9, "hybrid model failed the long-range recall task"
    assert attn_ms[-1] / ssm_ms[-1] > attn_ms[0] / ssm_ms[0], \
        "attention should grow faster than SSM with length"
    print("OK: hybrid Mamba-Attention learns long-range recall; SSM stays linear vs attention's quadratic.")


if __name__ == "__main__":
    main()
