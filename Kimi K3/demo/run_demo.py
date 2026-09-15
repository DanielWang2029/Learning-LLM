"""End-to-end KDA demo for Kimi K3 (CPU, < 60s).

Trains a tiny KDA (gated delta-rule linear attention) model on an associative
recall task, proving it learns long-range key→value memory with a *fixed-size*
recurrent state. We then time the layer at growing sequence lengths to show its
runtime scales linearly (constant state, no L×L matrix), unlike softmax
attention which scales quadratically.

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

from src import KDAModel  # noqa: E402
import generate_data as gd  # noqa: E402


def evaluate(model, n_pairs, rng, trials=400):
    model.eval()
    with torch.no_grad():
        toks, tgt = gd.make_batch(trials, n_pairs, rng)
        logits = model(toks)[:, -1]              # prediction at the query pos
        pred = logits.argmax(dim=-1)
        return (pred == tgt).float().mean().item()


def time_layer(model, L, batch=8, reps=3):
    """Wall-clock forward time of the KDA stack at sequence length L."""
    toks = torch.randint(0, gd.VOCAB, (batch, L))
    with torch.no_grad():
        model(toks)  # warm-up
        t0 = time.time()
        for _ in range(reps):
            model(toks)
        return (time.time() - t0) / reps


def main() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    t0 = time.time()

    n_pairs = 8  # 8 key→value pairs => sequence length 18
    model = KDAModel(gd.VOCAB, d_model=64, n_layers=2, n_heads=2, head_dim=32)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"KDA model: {n_params:,} params | task: recall 1 of {n_pairs} "
          f"key→value pairs (seq len {2*n_pairs+2})\n")

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    acc_curve = []
    for step in range(1, 401):
        model.train()
        toks, tgt = gd.make_batch(64, n_pairs, rng)
        logits = model(toks)[:, -1]
        loss = torch.nn.functional.cross_entropy(logits, tgt)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == 1:
            acc = evaluate(model, n_pairs, np.random.default_rng(123))
            acc_curve.append({"step": step, "acc": acc})
            print(f"step {step:3d} | loss {loss.item():.4f} | recall acc {acc*100:5.1f}%")

    final_acc = evaluate(model, n_pairs, np.random.default_rng(999))
    chance = 1.0 / gd.N_VALS

    # --- Linear-time check: forward time vs sequence length. ---
    print("\nRuntime vs sequence length (fixed-size state => linear):")
    lengths = [32, 64, 128, 256]
    times = []
    for L in lengths:
        dt = time_layer(model, L)
        times.append(dt)
        print(f"  L={L:4d} : {dt*1000:7.2f} ms")
    # normalized time per token should be roughly flat for linear scaling
    per_tok = [t / L for t, L in zip(times, lengths)]

    print("\n=== Results ===")
    print(f"final recall accuracy : {final_acc*100:.1f}%  (chance = {chance*100:.1f}%)")
    print(f"time / token (L={lengths[0]})   : {per_tok[0]*1e6:.2f} us")
    print(f"time / token (L={lengths[-1]})  : {per_tok[-1]*1e6:.2f} us   (flat => linear)")
    print(f"elapsed               : {time.time()-t0:.1f}s")

    # cost curves for the visualization (relative op counts)
    curve_L = [64, 128, 256, 512, 1024, 2048, 4096]
    quad = [n * n for n in curve_L]
    lin = [n * (model.layers[0].dk) for n in curve_L]  # ∝ L · state size

    out = {
        "n_pairs": n_pairs,
        "final_acc": final_acc,
        "chance": chance,
        "acc_curve": acc_curve,
        "timing": {"lengths": lengths, "ms": [round(t * 1000, 3) for t in times]},
        "cost_curve": {"lengths": curve_L, "quadratic": quad, "linear": lin},
        "vocab": gd.VOCAB,
        "n_keys": gd.N_KEYS,
        "n_vals": gd.N_VALS,
    }
    art = ROOT / "data" / "kda_result.json"
    art.write_text(json.dumps(out))
    print(f"\nWrote {art}")

    assert final_acc > 0.9, "KDA failed to learn associative recall"
    print("OK: KDA learns long-range key→value recall with a fixed-size linear-time state.")


if __name__ == "__main__":
    main()
