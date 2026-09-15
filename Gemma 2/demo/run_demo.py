"""Gemma 2 in miniature, end-to-end on CPU (arXiv 2408.00118).

What this demonstrates:

  1. ALTERNATING local (sliding-window) and global attention layers — we print
     the per-layer schedule and the two mask shapes.
  2. GQA + RMSNorm — reported in the model summary.
  3. LOGIT SOFT-CAPPING — we train two identical tiny models, one WITH capping
     and one WITHOUT, on a deterministic repeat task.  Without capping the raw
     logits grow without bound; capping pins them inside (-cap, cap) and keeps
     training stable, while reaching the same accuracy.

Run with:  python demo/run_demo.py       (finishes in well under a minute on CPU)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))

torch.set_num_threads(1)  # shared box: avoid thread oversubscription

from src import (  # noqa: E402
    Gemma2Config,
    Gemma2Model,
    build_causal_mask,
    build_sliding_window_mask,
    soft_cap,
)
from generate_data import make_dataset, VOCAB_SIZE, HALF, SEP  # noqa: E402

DATA_DIR = ROOT / "data"
SEED = 0
STEPS = 260
BATCH = 32
LR = 3e-3


def banner(t: str) -> None:
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def batches(data, batch_size, gen):
    data = torch.tensor(data, dtype=torch.long)
    n = data.size(0)
    while True:
        idx = torch.randint(0, n, (batch_size,), generator=gen)
        yield data[idx]


def next_token_batch(seqs):
    """Inputs are all-but-last; targets are all-but-first (shifted)."""
    return seqs[:, :-1], seqs[:, 1:]


@torch.no_grad()
def copy_accuracy(model, test):
    """Accuracy on the *copied* region only (positions after the SEP token)."""
    model.eval()
    x, y = next_token_batch(test)
    logits = model(x)
    pred = logits.argmax(-1)
    # The copied region is the final HALF target positions.
    region = slice(-HALF, None)
    correct = (pred[:, region] == y[:, region]).float().mean().item()
    return correct


def train_model(use_soft_cap, train, test, curve_key):
    cfg = Gemma2Config(vocab_size=VOCAB_SIZE, use_soft_cap=use_soft_cap)
    torch.manual_seed(SEED)
    model = Gemma2Model(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(SEED)
    stream = batches(train, BATCH, gen)
    test_t = torch.tensor(test, dtype=torch.long)

    curve = {"step": [], "loss": [], "acc": [], "max_out_logit": []}
    first_loss = None
    for step in range(1, STEPS + 1):
        model.train()
        seqs = next(stream)
        x, y = next_token_batch(seqs)
        logits = model(x)                          # already soft-capped if enabled
        loss = loss_fn(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

        if first_loss is None:
            first_loss = loss.item()

        if step % 20 == 0 or step == 1:
            acc = copy_accuracy(model, test_t)
            # Magnitude of the OUTPUT logits the loss actually sees. With capping
            # this can never exceed final_cap; without capping it is unbounded.
            out_max = float(logits.detach().abs().max().item())
            curve["step"].append(step)
            curve["loss"].append(round(loss.item(), 4))
            curve["acc"].append(round(acc, 4))
            curve["max_out_logit"].append(round(out_max, 3))
            tag = "cap" if use_soft_cap else "no-cap"
            print(f"  [{tag:>6}] step {step:3d} | loss {loss.item():8.3f} "
                  f"| copy-acc {acc*100:5.1f}% | max|output logit| "
                  f"{out_max:7.2f}")
    return model, cfg, curve, first_loss


def main() -> None:
    t0 = time.time()
    print("Gemma 2 in miniature — alternating attention + logit soft-capping")

    train = make_dataset(512, seed=0)
    test = make_dataset(64, seed=1)

    # --- Model summary (GQA + RMSNorm + alternating layers) ------------------
    cfg = Gemma2Config(vocab_size=VOCAB_SIZE)
    probe = Gemma2Model(cfg)
    banner("MODEL — GQA, RMSNorm, and ALTERNATING local/global attention")
    schedule = ["local (sliding-window)" if loc else "global (full causal)"
                for loc in probe.layer_is_local]
    print(f"  params: {probe.num_params():,}")
    print(f"  query heads: {cfg.n_heads}  kv heads: {cfg.n_kv_heads}  "
          f"(GQA: {cfg.n_rep} query heads per kv head)")
    print(f"  sliding window: {cfg.sliding_window} tokens")
    print("  layer schedule:")
    for i, s in enumerate(schedule):
        print(f"    layer {i}: {s}")

    # Show the two mask shapes on a short sequence.
    demo_len = 10
    local_mask = build_sliding_window_mask(demo_len, cfg.sliding_window)
    global_mask = build_causal_mask(demo_len)

    def show_mask(name, m):
        print(f"\n  {name} (rows=query, cols=key; # = attend):")
        for row in m.int().tolist():
            print("    " + " ".join("#" if v else "." for v in row))

    show_mask("LOCAL sliding-window mask", local_mask)
    show_mask("GLOBAL causal mask", global_mask)

    # --- Soft-capping ablation ----------------------------------------------
    banner("SOFT-CAPPING ABLATION — train WITH vs WITHOUT logit soft-capping")
    print("Both models are identical except for logit soft-capping.\n")
    print(" Training WITH soft-capping (attn cap=50, final cap=30):")
    model_cap, cfg_cap, curve_cap, first_loss_cap = train_model(True, train, test, "cap")
    print("\n Training WITHOUT soft-capping:")
    model_nocap, cfg_nocap, curve_nocap, first_loss_nocap = train_model(
        False, train, test, "nocap")

    acc_cap = copy_accuracy(model_cap, torch.tensor(test, dtype=torch.long))
    acc_nocap = copy_accuracy(model_nocap, torch.tensor(test, dtype=torch.long))
    out_max_cap = max(curve_cap["max_out_logit"])
    out_max_nocap = max(curve_nocap["max_out_logit"])

    banner("RESULTS")
    print(f"  WITH    soft-cap: initial loss {first_loss_cap:7.2f} | "
          f"copy-acc {acc_cap*100:5.1f}% | peak |output logit| {out_max_cap:7.2f} "
          f"(<= final cap {cfg_cap.final_logit_softcap:.0f})")
    print(f"  WITHOUT soft-cap: initial loss {first_loss_nocap:7.2f} | "
          f"copy-acc {acc_nocap*100:5.1f}% | peak |output logit| {out_max_nocap:7.2f} "
          f"(unbounded)")
    print(f"\n  -> soft-capping (a) keeps output logits inside "
          f"[-{cfg_cap.final_logit_softcap:.0f}, {cfg_cap.final_logit_softcap:.0f}] "
          f"vs {out_max_nocap:.0f} uncapped, and (b) tames the initial-loss spike "
          f"({first_loss_nocap:.0f} -> {first_loss_cap:.0f}), so training converges "
          f"({acc_cap*100:.0f}% vs {acc_nocap*100:.0f}%).")

    # --- Soft-cap transfer curve for the visualization ----------------------
    xs = torch.linspace(-120, 120, 121)
    ys30 = soft_cap(xs, 30.0)
    ys50 = soft_cap(xs, 50.0)

    # --- Persist a small JSON for the HTML ----------------------------------
    DATA_DIR.mkdir(exist_ok=True)
    payload = {
        "config": {
            "d_model": cfg.d_model, "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads, "n_kv_heads": cfg.n_kv_heads,
            "n_rep": cfg.n_rep, "sliding_window": cfg.sliding_window,
            "attn_cap": cfg.attn_logit_softcap, "final_cap": cfg.final_logit_softcap,
            "params": probe.num_params(),
        },
        "layer_schedule": ["local" if loc else "global" for loc in probe.layer_is_local],
        "local_mask": local_mask.int().tolist(),
        "global_mask": global_mask.int().tolist(),
        "sliding_window_demo_len": demo_len,
        "softcap_curve": {
            "x": [round(v, 2) for v in xs.tolist()],
            "cap30": [round(v, 3) for v in ys30.tolist()],
            "cap50": [round(v, 3) for v in ys50.tolist()],
        },
        "training": {"with_cap": curve_cap, "without_cap": curve_nocap},
        "results": {
            "acc_with_cap": acc_cap, "acc_without_cap": acc_nocap,
            "peak_out_logit_with_cap": out_max_cap,
            "peak_out_logit_without_cap": out_max_nocap,
            "initial_loss_with_cap": round(first_loss_cap, 3),
            "initial_loss_without_cap": round(first_loss_nocap, 3),
        },
    }
    out = DATA_DIR / "gemma2_results.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n  wrote {out}")
    print(f"  total time: {time.time()-t0:.1f}s")

    # --- Sanity asserts ------------------------------------------------------
    assert acc_cap > 0.9, f"capped model failed to learn ({acc_cap:.2f})"
    assert out_max_cap <= cfg_cap.final_logit_softcap + 1e-3, (
        "capped output logits must stay within the final cap")
    assert out_max_nocap > cfg_cap.final_logit_softcap, (
        "uncapped output logits should exceed the cap")
    assert first_loss_nocap > first_loss_cap, (
        "soft-capping should reduce the initial-loss spike")
    print("\nOK: alternating attention works; soft-capping bounds the output "
          "logits and stabilizes training.")


if __name__ == "__main__":
    main()
