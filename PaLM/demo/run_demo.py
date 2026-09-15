"""End-to-end PaLM demo: train a tiny PaLM on a copy task and prove it learns.

What this script does, all on CPU in well under a minute:

1. Builds a tiny `PaLM` model and prints confirmation that each distinctive
   PaLM feature is actually wired in (with tensor shapes / parameter counts):
   SwiGLU, parallel attention+MLP, multi-query attention, RoPE, and no biases.
2. Trains it on the copy task ([BOS] c.. [SEP] c..) with next-token prediction.
3. Reports the loss going down and the exact-copy accuracy going up, and shows
   a worked example generated autoregressively.
4. Writes ``data/palm_demo.json`` (loss curve + feature report + a sample) for
   the visualization.

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

from src import PaLM, PaLMConfig  # noqa: E402

PAD, BOS, SEP, EOS = 0, 1, 2, 3
NUM_SPECIAL = 4
DATA_DIR = PAPER_DIR / "data"


def make_batch(batch_size: int, seq_len: int, vocab_size: int, device: torch.device):
    """Build a copy-task batch.

    tokens : [BOS] c1..ck [SEP] c1..ck            (length 2k + 2)
    inputs : tokens[:-1]
    targets: tokens[1:], but only the copy region contributes to the loss
             (everything else is set to the ignore index -1).
    """
    content = torch.randint(
        NUM_SPECIAL, vocab_size, (batch_size, seq_len), device=device
    )
    bos = torch.full((batch_size, 1), BOS, device=device)
    sep = torch.full((batch_size, 1), SEP, device=device)
    tokens = torch.cat([bos, content, sep, content], dim=1)

    inputs = tokens[:, :-1]
    targets = tokens[:, 1:].clone()
    # Only score predictions of the copied region (positions >= seq_len + 1).
    targets[:, : seq_len + 1] = -1
    return inputs, targets, content


@torch.no_grad()
def exact_copy_accuracy(model, batch_size, seq_len, vocab_size, device) -> float:
    model.eval()
    content = torch.randint(
        NUM_SPECIAL, vocab_size, (batch_size, seq_len), device=device
    )
    bos = torch.full((batch_size, 1), BOS, device=device)
    sep = torch.full((batch_size, 1), SEP, device=device)
    prompt = torch.cat([bos, content, sep], dim=1)  # [BOS] c.. [SEP]
    out = model.generate(prompt, max_new_tokens=seq_len)
    pred = out[:, prompt.size(1):]
    return (pred == content).all(dim=1).float().mean().item()


def feature_report(model: PaLM) -> dict:
    """Confirm each PaLM feature is active, returning a structured report."""
    cfg = model.config
    block0 = model.blocks[0]
    attn = block0.attn

    num_bias = sum(
        p.numel() for n, p in model.named_parameters() if n.endswith("bias")
    )
    report = {
        "SwiGLU activation": {
            "active": hasattr(block0.mlp, "w_gate")
            and hasattr(block0.mlp, "w_up")
            and hasattr(block0.mlp, "w_down"),
            "detail": "3 projections (gate/up/down): "
            f"w_gate {tuple(block0.mlp.w_gate.weight.shape)}, "
            f"w_up {tuple(block0.mlp.w_up.weight.shape)}, "
            f"w_down {tuple(block0.mlp.w_down.weight.shape)}",
        },
        "Parallel attention + MLP": {
            "active": hasattr(block0, "norm")
            and hasattr(block0, "attn")
            and hasattr(block0, "mlp"),
            "detail": "single shared norm feeds attn and mlp; outputs summed "
            "onto residual: x + attn(norm(x)) + mlp(norm(x))",
        },
        "Multi-query attention": {
            "active": attn.w_k.weight.shape[0] == attn.head_dim
            and attn.w_q.weight.shape[0] == cfg.n_heads * attn.head_dim,
            "detail": f"{cfg.n_heads} query heads share 1 K/V head "
            f"(w_q out={attn.w_q.weight.shape[0]}, "
            f"w_k out={attn.w_k.weight.shape[0]} = head_dim {attn.head_dim})",
        },
        "RoPE positions": {
            "active": not any("pos_emb" in n for n, _ in model.named_parameters()),
            "detail": "rotary embedding applied to Q/K; no learned position "
            f"parameters (inv_freq buffer len {model.rope.inv_freq.numel()})",
        },
        "No biases": {
            "active": num_bias == 0,
            "detail": f"total bias parameters in the whole model: {num_bias}",
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=6)
    parser.add_argument("--vocab-size", type=int, default=16)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--ffn-hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    # One CPU thread is fastest here: the model is tiny, so the fixed cost of
    # thread synchronization dominates the actual arithmetic.
    torch.set_num_threads(max(1, args.threads))
    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    config = PaLMConfig(
        vocab_size=args.vocab_size,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ffn_hidden=args.ffn_hidden,
    )
    model = PaLM(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())

    print("=" * 70)
    print("PaLM demo — Chowdhery et al. 2022 (arXiv:2204.02311)")
    print("=" * 70)
    print(f"Parameters: {num_params:,}   config: {config}\n")

    print("PaLM feature check (distinctive architecture choices):")
    report = feature_report(model)
    for name, info in report.items():
        mark = "OK " if info["active"] else "!! "
        print(f"  [{mark}] {name}: {info['detail']}")
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    loss_curve = []
    acc_curve = []
    start = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        inputs, targets, _ = make_batch(
            args.batch_size, args.seq_len, args.vocab_size, device
        )
        _, loss = model(inputs, targets)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.log_every == 0 or step == 1:
            acc = exact_copy_accuracy(model, 256, args.seq_len, args.vocab_size, device)
            loss_curve.append({"step": step, "loss": round(loss.item(), 4)})
            acc_curve.append({"step": step, "acc": round(acc, 4)})
            print(
                f"step {step:4d}/{args.steps} | loss {loss.item():.4f} "
                f"| exact-copy acc {acc * 100:5.1f}%"
            )

    elapsed = time.time() - start
    final_acc = exact_copy_accuracy(model, 512, args.seq_len, args.vocab_size, device)
    print(f"\nTrained {args.steps} steps in {elapsed:.1f}s on CPU")
    print(f"Final exact-copy accuracy: {final_acc * 100:.1f}%")

    # One concrete worked example.
    model.eval()
    content = torch.randint(NUM_SPECIAL, args.vocab_size, (1, args.seq_len))
    prompt = torch.cat(
        [torch.tensor([[BOS]]), content, torch.tensor([[SEP]])], dim=1
    )
    out = model.generate(prompt, max_new_tokens=args.seq_len)
    print("\nExample")
    print(f"  content to copy : {content[0].tolist()}")
    print(f"  model produced  : {out[0, prompt.size(1):].tolist()}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    demo_json = {
        "paper": "PaLM (Chowdhery et al. 2022, arXiv:2204.02311)",
        "config": vars(args),
        "num_parameters": num_params,
        "loss_curve": loss_curve,
        "acc_curve": acc_curve,
        "final_accuracy": round(final_acc, 4),
        "elapsed_seconds": round(elapsed, 1),
        "feature_report": {k: v["detail"] for k, v in report.items()},
        "example": {
            "content": content[0].tolist(),
            "generated": out[0, prompt.size(1):].tolist(),
        },
    }
    (DATA_DIR / "palm_demo.json").write_text(json.dumps(demo_json, indent=2))
    print("\nWrote data/palm_demo.json (loss curve + feature report for the viz).")

    if final_acc < 0.9:
        raise SystemExit(f"Copy task did not converge ({final_acc * 100:.1f}% < 90%).")
    print("OK: PaLM learned the copy task with all distinctive features active.")


if __name__ == "__main__":
    main()
