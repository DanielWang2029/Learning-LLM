"""End-to-end Llama 3 demo (CPU, well under a minute).

Two parts, matching the paper's two headline stories:

1. ARCHITECTURE. Train a tiny Llama 3 model (RMSNorm + RoPE@500k +
   Grouped-Query Attention + SwiGLU) on a toy copy task and show the loss fall
   and exact-copy accuracy hit ~100%. A feature-check confirms GQA is really
   active by reporting the KV-cache reduction factor (n_heads / n_kv_heads).

2. TOKENIZER. Train from-scratch byte-level BPE tokenizers at several vocabulary
   sizes on a small English corpus, then encode held-out text with each. Larger
   vocabularies merge more subwords, so the SAME text is encoded in FEWER tokens
   — exactly the effect behind Llama 3's move from a 32K to a 128K vocabulary.
   We report tokens and the compression ratio (bytes/token) for each size.

Writes ``data/llama3_demo.json`` (loss curve + tokenizer table) for the viz.

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

PAPER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PAPER_DIR))

from src import ByteBPETokenizer, Llama3, Llama3Config  # noqa: E402

sys.path.insert(0, str(PAPER_DIR / "data"))
from generate_demo_data import build_corpus  # noqa: E402

DATA_DIR = PAPER_DIR / "data"
PAD, BOS, SEP = 0, 1, 2
NUM_SPECIAL = 3

HELDOUT_TEXT = (
    "A larger vocabulary lets the tokenizer represent common words as single "
    "tokens, so the same sentence about language modeling costs fewer tokens."
)


def make_batch(batch_size, seq_len, vocab_size, device):
    content = torch.randint(NUM_SPECIAL, vocab_size, (batch_size, seq_len), device=device)
    bos = torch.full((batch_size, 1), BOS, device=device)
    sep = torch.full((batch_size, 1), SEP, device=device)
    tokens = torch.cat([bos, content, sep, content], dim=1)
    inputs = tokens[:, :-1]
    targets = tokens[:, 1:].clone()
    targets[:, : seq_len + 1] = -1  # score only the copied region
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


def train_architecture(args):
    torch.manual_seed(args.seed)
    cfg = Llama3Config(
        vocab_size=args.vocab_size, dim=args.dim, n_layers=args.n_layers,
        n_heads=args.n_heads, n_kv_heads=args.n_kv_heads,
    )
    model = Llama3(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}   config: {cfg}")
    print(f"Feature-check | RMSNorm: on | RoPE base: {cfg.rope_base:g} | "
          f"SwiGLU: on | GQA: {cfg.n_heads} query heads share {cfg.n_kv_heads} "
          f"K/V heads  ->  KV-cache reduction x{cfg.n_heads // cfg.n_kv_heads}\n")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    curve = []
    start = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        inputs, targets = make_batch(args.batch_size, args.seq_len, args.vocab_size, "cpu")
        _, loss = model(inputs, targets)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.log_every == 0 or step == 1:
            acc = exact_copy_accuracy(model, 256, args.seq_len, args.vocab_size, "cpu")
            curve.append({"step": step, "loss": round(loss.item(), 4), "acc": round(acc, 4)})
            print(f"  step {step:4d}/{args.steps} | loss {loss.item():.4f} | "
                  f"exact-copy acc {acc*100:5.1f}%")
    elapsed = time.time() - start
    final_acc = exact_copy_accuracy(model, 512, args.seq_len, args.vocab_size, "cpu")
    print(f"\nArchitecture: final exact-copy accuracy {final_acc*100:.1f}% "
          f"(trained {args.steps} steps in {elapsed:.1f}s)")
    return cfg, n_params, curve, final_acc


def tokenizer_study(vocab_sizes):
    print("\n" + "=" * 70)
    print("Tokenizer efficiency: same text, growing BPE vocabulary")
    print("=" * 70)
    corpus = build_corpus()
    print(f"Training corpus: {len(corpus.encode('utf-8'))} bytes")
    print(f"Held-out text  : {len(HELDOUT_TEXT.encode('utf-8'))} bytes "
          f"({HELDOUT_TEXT[:48]}...)\n")

    rows = []
    baseline_tokens = None
    print(f"  {'vocab_size':>10} | {'tokens':>6} | {'bytes/token':>11} | {'vs 256-byte':>11}")
    print("  " + "-" * 48)
    for vs in vocab_sizes:
        tok = ByteBPETokenizer().train(corpus, vocab_size=vs)
        ids = tok.encode(HELDOUT_TEXT)
        n_tokens = len(ids)
        ratio = tok.compression_ratio(HELDOUT_TEXT)
        if baseline_tokens is None:
            baseline_tokens = n_tokens
        reduction = 100.0 * (1 - n_tokens / baseline_tokens)
        rows.append({"vocab_size": tok.vocab_size, "tokens": n_tokens,
                     "bytes_per_token": round(ratio, 3),
                     "reduction_vs_bytes_pct": round(reduction, 1)})
        print(f"  {tok.vocab_size:>10} | {n_tokens:>6} | {ratio:>11.3f} | "
              f"{reduction:>9.1f}%")

    best, base = rows[-1], rows[0]
    print(f"\nGrowing the vocabulary {base['vocab_size']} -> {best['vocab_size']} "
          f"cut the held-out text from {base['tokens']} to {best['tokens']} tokens "
          f"({best['reduction_vs_bytes_pct']:.1f}% fewer), raising compression from "
          f"{base['bytes_per_token']:.2f} to {best['bytes_per_token']:.2f} bytes/token.")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=6)
    p.add_argument("--vocab-size", type=int, default=32)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-kv-heads", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print("=" * 70)
    print("Llama 3 demo — The Llama 3 Herd of Models (Meta 2024, arXiv:2407.21783)")
    print("=" * 70)
    cfg, n_params, curve, final_acc = train_architecture(args)
    token_rows = tokenizer_study([256, 384, 512, 640])

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    demo_json = {
        "paper": "The Llama 3 Herd of Models (Meta 2024, arXiv:2407.21783)",
        "architecture": {
            "num_parameters": n_params,
            "dim": cfg.dim, "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads, "n_kv_heads": cfg.n_kv_heads,
            "kv_cache_reduction": cfg.n_heads // cfg.n_kv_heads,
            "rope_base": cfg.rope_base,
            "loss_curve": curve,
            "final_accuracy": round(final_acc, 4),
        },
        "tokenizer": {
            "heldout_text": HELDOUT_TEXT,
            "heldout_bytes": len(HELDOUT_TEXT.encode("utf-8")),
            "rows": token_rows,
        },
    }
    (DATA_DIR / "llama3_demo.json").write_text(json.dumps(demo_json, indent=2))
    print("\nWrote data/llama3_demo.json (loss curve + tokenizer table for the viz).")

    if final_acc < 0.9:
        raise SystemExit(f"Architecture did not converge ({final_acc*100:.1f}% < 90%).")
    if token_rows[-1]["tokens"] >= token_rows[0]["tokens"]:
        raise SystemExit("Tokenizer study failed to show fewer tokens with a larger vocab.")
    print("OK: Llama 3 recipe learned the copy task; larger vocab compressed the text.")


if __name__ == "__main__":
    main()
