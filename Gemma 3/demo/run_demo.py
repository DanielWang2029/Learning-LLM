"""End-to-end Gemma 3 demo: 5:1 local:global attention (CPU, <60s).

Trains three tiny transformers on the same long-context retrieval task:

  1. 5:1 local:global  (Gemma 3's pattern)
  2. all-global        (accurate but expensive — full attention everywhere)
  3. all-local         (cheap but can't route long-range info)

It verifies the 5:1 layer pattern, then reports each model's accuracy and its
attention-memory footprint (allowed attention-score entries). The 5:1 model
matches all-global accuracy while using far less attention memory; all-local is
cheap but fails the long-range retrieval.

Run with:  python demo/run_demo.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(1)  # shared CPU box: avoid thread oversubscription

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import (TinyTransformer, VOCAB_SIZE, N_VALUE, layer_pattern,
                 make_batch, describe_token)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def build(pattern, seq_len, window, qk_norm=True):
    return TinyTransformer(VOCAB_SIZE, N_VALUE, seq_len, n_layers=len(pattern),
                           window=window, pattern=pattern, qk_norm=qk_norm)


def train(model, steps, seq_len, batch, lr, gen):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    for _ in range(steps):
        tokens, qpos, target, _ = make_batch(batch, seq_len, gen)
        logits = model(tokens, qpos)
        loss = loss_fn(logits, target)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def evaluate(model, seq_len, n, gen):
    """Overall accuracy + accuracy bucketed by how far the payload is from the end."""
    model.eval()
    tokens, qpos, target, payload_pos = make_batch(n, seq_len, gen)
    pred = model(tokens, qpos).argmax(-1)
    correct = (pred == target)
    dist = (qpos - payload_pos)
    far = dist > (seq_len // 2)
    acc = correct.float().mean().item()
    acc_far = correct[far].float().mean().item() if far.any() else float("nan")
    acc_near = correct[~far].float().mean().item() if (~far).any() else float("nan")
    return acc, acc_near, acc_far


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    p_53 = layer_pattern(args.n_layers, ratio=5)
    p_global = ["global"] * args.n_layers
    p_local = ["local"] * args.n_layers

    print(f"Task: long-context retrieval, sequence length {args.seq_len}, "
          f"local window radius {args.window}")
    print(f"5:1 layer pattern ({args.n_layers} layers): {p_53}")
    n_global = p_53.count("global")
    assert p_53.count("local") == args.n_layers - n_global
    print(f"  -> {p_53.count('local')} local : {n_global} global "
          f"(ratio {p_53.count('local')}:{n_global}), QK-norm on\n")

    configs = {"5:1 local:global": p_53, "all-global": p_global, "all-local": p_local}
    results = {}
    models = {}
    t0 = time.time()
    for name, pattern in configs.items():
        gen = torch.Generator().manual_seed(args.seed)  # same data stream per model
        model = build(pattern, args.seq_len, args.window)
        train(model, args.steps, args.seq_len, args.batch, args.lr, gen)
        acc, acc_near, acc_far = evaluate(model, args.seq_len, args.n_test, gen)
        entries = model.attention_entries()
        models[name] = model
        results[name] = {"pattern": pattern, "accuracy": round(acc, 4),
                         "acc_near": round(acc_near, 4), "acc_far": round(acc_far, 4),
                         "attn_entries": entries}
        print(f"  {name:18s} | acc {acc*100:5.1f}% (near {acc_near*100:4.0f}%, "
              f"far {acc_far*100:4.0f}%) | attn entries {entries:6d}")

    g_entries = results["all-global"]["attn_entries"]
    for name in results:
        results[name]["memory_vs_global"] = round(
            results[name]["attn_entries"] / g_entries, 3)
    saving = 1 - results["5:1 local:global"]["attn_entries"] / g_entries
    print(f"\n5:1 uses {results['5:1 local:global']['memory_vs_global']*100:.0f}% of "
          f"all-global attention memory ({saving*100:.0f}% saved), "
          f"at {results['5:1 local:global']['accuracy']*100:.0f}% accuracy "
          f"vs all-global {results['all-global']['accuracy']*100:.0f}%.")

    # Show a concrete retrieval example using the already-trained 5:1 model.
    gen = torch.Generator().manual_seed(args.seed + 123)
    model = models["5:1 local:global"]
    model.eval()
    tokens, qpos, target, payload_pos = make_batch(1, args.seq_len, gen)
    pred = int(model(tokens, qpos).argmax(-1)[0])
    p = int(payload_pos[0])
    print(f"\nExample (payload at position {p}, {int(qpos[0]) - p} tokens from query):")
    print(f"  ...{[describe_token(int(x)) for x in tokens[0, max(0,p-1):p+2]]}... "
          f"QUERY@{int(qpos[0])}")
    print(f"  true value = {int(target[0])}, 5:1 model predicted = {pred} "
          f"{'OK' if pred == int(target[0]) else 'WRONG'}")

    out = {
        "config": {"seq_len": args.seq_len, "n_layers": args.n_layers,
                   "window": args.window, "n_test": args.n_test,
                   "seconds": round(time.time() - t0, 1)},
        "results": results,
        "example": {"payload_pos": p, "distance": int(qpos[0]) - p,
                    "true": int(target[0]), "pred": pred},
    }
    DATA_DIR.mkdir(exist_ok=True)
    out_path = DATA_DIR / "gemma3_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")

    r = results
    ok = (r["5:1 local:global"]["accuracy"] > 0.9
          and r["5:1 local:global"]["attn_entries"] < g_entries
          and r["5:1 local:global"]["acc_far"] - r["all-local"]["acc_far"] > 0.2)
    if not ok:
        raise SystemExit("5:1 pattern did not match global accuracy at lower memory.")
    print("\nOK: the 5:1 local:global pattern retrieves long-range info like "
          "all-global, at a fraction of the attention memory — and beats all-local.")


if __name__ == "__main__":
    main()
