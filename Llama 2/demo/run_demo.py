"""End-to-end Llama 2 demo: train with GQA, and quantify the KV-cache savings.

Runs entirely on CPU in well under a minute:

1. Trains a tiny Llama 2 with Grouped-Query Attention (n_kv_heads=2 shared
   across 8 query heads) on a toy copy task, reaching ~100% exact-copy accuracy.
2. Confirms GQA works across the whole spectrum: retrains the same model as
   MHA (n_kv=8), GQA (n_kv=2) and MQA (n_kv=1) and reports that all three learn
   — while their KV caches differ dramatically.
3. Prints the KV-cache memory for MHA vs GQA vs MQA, both for the demo config
   and for a realistic Llama-2-70B-scale config, showing the savings GQA buys.
4. Writes ``data/llama2_demo.json`` (loss curve + KV-cache tables) for the viz.

Run with:  python demo/run_demo.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

PAPER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PAPER_DIR))

from src import Llama2, Llama2Config, kv_cache_bytes  # noqa: E402

PAD, BOS, SEP, EOS = 0, 1, 2, 3
NUM_SPECIAL = 4
DATA_DIR = PAPER_DIR / "data"


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.2f} {unit}"
        n /= 1024


def make_batch(batch_size, seq_len, vocab_size, device):
    content = torch.randint(NUM_SPECIAL, vocab_size, (batch_size, seq_len), device=device)
    bos = torch.full((batch_size, 1), BOS, device=device)
    sep = torch.full((batch_size, 1), SEP, device=device)
    tokens = torch.cat([bos, content, sep, content], dim=1)
    inputs = tokens[:, :-1]
    targets = tokens[:, 1:].clone()
    targets[:, : seq_len + 1] = -1
    return inputs, targets


@torch.no_grad()
def exact_copy_accuracy(model, batch_size, seq_len, vocab_size, device):
    model.eval()
    content = torch.randint(NUM_SPECIAL, vocab_size, (batch_size, seq_len), device=device)
    bos = torch.full((batch_size, 1), BOS, device=device)
    sep = torch.full((batch_size, 1), SEP, device=device)
    prompt = torch.cat([bos, content, sep], dim=1)
    out = model.generate(prompt, max_new_tokens=seq_len)
    return (out[:, prompt.size(1):] == content).all(dim=1).float().mean().item()


def train(config, steps, args, record_curve=False, seed=0):
    torch.manual_seed(seed)
    model = Llama2(config)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    curve = []
    loss = torch.tensor(0.0)
    for step in range(1, steps + 1):
        model.train()
        inputs, targets = make_batch(args.batch_size, args.seq_len, args.vocab_size, "cpu")
        _, loss = model(inputs, targets)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if record_curve and (step % args.log_every == 0 or step == 1):
            acc = exact_copy_accuracy(model, 256, args.seq_len, args.vocab_size, "cpu")
            curve.append({"step": step, "loss": round(loss.item(), 4), "acc": round(acc, 4)})
            print(f"step {step:4d}/{steps} | loss {loss.item():.4f} | exact-copy acc {acc*100:5.1f}%")
    final_acc = exact_copy_accuracy(model, 512, args.seq_len, args.vocab_size, "cpu")
    return model, curve, loss.item(), final_acc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--regime-steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=6)
    parser.add_argument("--vocab-size", type=int, default=16)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-kv-heads", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.threads))
    base = dict(vocab_size=args.vocab_size, dim=args.dim, n_layers=args.n_layers, n_heads=args.n_heads)

    print("=" * 70)
    print("Llama 2 demo — Touvron et al. 2023 (arXiv:2307.09288)")
    print("=" * 70)
    full_cfg = Llama2Config(**base, n_kv_heads=args.n_kv_heads)
    model = Llama2(full_cfg)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}   config: {full_cfg}")
    print(f"Attention: {args.n_heads} query heads share {args.n_kv_heads} K/V heads "
          f"({args.n_heads // args.n_kv_heads} query heads per group) -> GQA\n")

    print("Training Llama 2 with Grouped-Query Attention on the copy task:")
    start = time.time()
    _, curve, floss, facc = train(full_cfg, args.steps, args, record_curve=True, seed=0)
    print(f"\nGQA model: final loss {floss:.4f} | exact-copy accuracy {facc*100:.1f}%")

    # ---- All three regimes learn; KV cache differs.
    head_dim = args.dim // args.n_heads
    print(f"\nAttention regimes (each retrained {args.regime_steps} steps, KV-cache at seq={args.seq_len}):")
    regimes = [("MHA", args.n_heads), ("GQA", args.n_kv_heads), ("MQA", 1)]
    regime_rows = []
    for name, nkv in regimes:
        _, _, _, acc = train(Llama2Config(**base, n_kv_heads=nkv), args.regime_steps, args, seed=0)
        kv = kv_cache_bytes(args.n_layers, nkv, head_dim, args.seq_len, batch_size=1)
        regime_rows.append({"regime": name, "n_kv_heads": nkv, "acc": round(acc, 4), "kv_bytes": kv})
        print(f"  {name} (n_kv={nkv}) | exact-copy acc {acc*100:5.1f}% | KV-cache {human_bytes(kv)}")

    # ---- Realistic Llama-2-70B-scale KV cache (the point of GQA).
    big = dict(n_layers=80, n_heads=64, head_dim=128, seq_len=4096, batch_size=1, dtype_bytes=2)
    print(f"\nKV-cache at Llama-2-70B scale (n_layers=80, n_heads=64, head_dim=128, "
          f"seq={big['seq_len']}, fp16):")
    big_regimes = [("MHA", 64), ("GQA", 8), ("MQA", 1)]
    big_rows = []
    mha_bytes = None
    for name, nkv in big_regimes:
        kv = kv_cache_bytes(big["n_layers"], nkv, big["head_dim"], big["seq_len"],
                            big["batch_size"], big["dtype_bytes"])
        if name == "MHA":
            mha_bytes = kv
        factor = mha_bytes / kv
        big_rows.append({"regime": name, "n_kv_heads": nkv, "kv_bytes": kv,
                         "reduction_vs_mha": round(factor, 1)})
        print(f"  {name} (n_kv={nkv:2d}) | KV-cache {human_bytes(kv):>10} | {factor:4.1f}x smaller than MHA")

    elapsed = time.time() - start
    print(f"\nTotal demo time: {elapsed:.1f}s on CPU")
    print("Takeaway: GQA matches MHA quality on the task while cutting the KV cache "
          f"{big_rows[1]['reduction_vs_mha']:.0f}x at 70B scale — cheaper long-context serving.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    demo_json = {
        "paper": "Llama 2 (Touvron et al. 2023, arXiv:2307.09288)",
        "config": vars(args),
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "loss_curve": curve,
        "final_loss": round(floss, 4),
        "final_accuracy": round(facc, 4),
        "elapsed_seconds": round(elapsed, 1),
        "regimes_demo": regime_rows,
        "big_config": {**big, "n_kv_heads_gqa": 8},
        "kv_cache_big": big_rows,
    }
    (DATA_DIR / "llama2_demo.json").write_text(json.dumps(demo_json, indent=2))
    print("\nWrote data/llama2_demo.json (loss curve + KV-cache tables for the viz).")

    if facc < 0.9:
        raise SystemExit(f"GQA model did not converge ({facc*100:.1f}% < 90%).")
    print("OK: Llama 2 (GQA) learned the copy task; KV-cache savings quantified.")


if __name__ == "__main__":
    main()
