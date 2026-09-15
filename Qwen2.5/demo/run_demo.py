"""End-to-end Qwen2.5 demo (CPU, a few seconds).

Two things happen:

1. FEATURE-CHECK. Build a tiny Qwen2.5 model and confirm each distinctive
   component is really wired in:
     - QKV bias on the attention projections (and NOT on the output projection),
     - untied input/output embeddings (separate weight matrices),
     - plus the standard RMSNorm + RoPE + GQA + SwiGLU ingredients.
   After training we re-check the QKV bias vectors have learned non-zero values
   and that the two embedding matrices have genuinely diverged — proof the
   features are active, not vestigial.

2. TRAIN. Learn a toy sequence-copy task and watch loss fall / exact-copy
   accuracy hit ~100%. A short ablation retrains with the QKV bias toggled off
   to show the switch is genuinely wired through the model (both converge on
   this easy task; the QKV bias's real benefit in the paper is extrapolation).

Writes ``data/qwen25_demo.json`` (feature report + loss curve) for the viz.

Run with:  python demo/run_demo.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

torch.set_num_threads(1)

PAPER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PAPER_DIR))

from src import Qwen25, Qwen25Config  # noqa: E402
from src.attention import QwenAttention  # noqa: E402
from src.layers import RMSNorm, RotaryEmbedding, SwiGLU  # noqa: E402

DATA_DIR = PAPER_DIR / "data"
PAD, BOS, SEP = 0, 1, 2
NUM_SPECIAL = 3


def make_batch(bs, seq_len, vocab, device):
    content = torch.randint(NUM_SPECIAL, vocab, (bs, seq_len), device=device)
    bos = torch.full((bs, 1), BOS, device=device)
    sep = torch.full((bs, 1), SEP, device=device)
    tokens = torch.cat([bos, content, sep, content], dim=1)
    inputs, targets = tokens[:, :-1], tokens[:, 1:].clone()
    targets[:, : seq_len + 1] = -1
    return inputs, targets


@torch.no_grad()
def exact_copy_accuracy(model, bs, seq_len, vocab, device):
    model.eval()
    content = torch.randint(NUM_SPECIAL, vocab, (bs, seq_len), device=device)
    bos = torch.full((bs, 1), BOS, device=device)
    sep = torch.full((bs, 1), SEP, device=device)
    prompt = torch.cat([bos, content, sep], dim=1)
    out = model.generate(prompt, max_new_tokens=seq_len)
    return (out[:, prompt.size(1):] == content).all(dim=1).float().mean().item()


def feature_check(model: Qwen25):
    """Inspect the built model and confirm each distinctive feature is present."""
    attn = model.layers[0].self_attn
    block = model.layers[0]
    checks = [
        ("QKV bias on Q/K/V projections", attn.q_proj.bias is not None
         and attn.k_proj.bias is not None and attn.v_proj.bias is not None),
        ("No bias on output projection", attn.o_proj.bias is None),
        ("Untied input/output embeddings", not model.embeddings_are_tied()),
        ("RMSNorm pre-normalization", isinstance(block.input_layernorm, RMSNorm)),
        ("RoPE rotary positions", isinstance(model.rope, RotaryEmbedding)),
        ("Grouped-Query Attention", attn.n_kv_heads < attn.n_heads),
        ("SwiGLU feed-forward", isinstance(block.mlp, SwiGLU)),
    ]
    return checks


def train(cfg, args, record=False, seed=0):
    torch.manual_seed(seed)
    model = Qwen25(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    curve = []
    for step in range(1, args.steps + 1):
        model.train()
        inputs, targets = make_batch(args.batch_size, args.seq_len, args.vocab_size, "cpu")
        _, loss = model(inputs, targets)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if record and (step % args.log_every == 0 or step == 1):
            acc = exact_copy_accuracy(model, 256, args.seq_len, args.vocab_size, "cpu")
            curve.append({"step": step, "loss": round(loss.item(), 4), "acc": round(acc, 4)})
            print(f"  step {step:4d}/{args.steps} | loss {loss.item():.4f} | "
                  f"exact-copy acc {acc*100:5.1f}%")
    acc = exact_copy_accuracy(model, 512, args.seq_len, args.vocab_size, "cpu")
    return model, curve, acc


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--ablation-steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=6)
    p.add_argument("--vocab-size", type=int, default=32)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--log-every", type=int, default=100)
    args = p.parse_args()

    print("=" * 70)
    print("Qwen2.5 demo — Qwen2.5 Technical Report (Alibaba 2024, arXiv:2412.15115)")
    print("=" * 70)

    cfg = Qwen25Config(vocab_size=args.vocab_size, dim=args.dim, n_layers=args.n_layers)
    model = Qwen25(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}   config: {cfg}\n")

    checks = feature_check(model)
    print("Feature-check (distinctive Qwen2.5 choices in bold):")
    for name, ok in checks:
        print(f"  [{'x' if ok else ' '}] {name}")
    tied = model.embeddings_are_tied()
    emb_params = model.embed_tokens.weight.numel()
    print(f"\n  Untied embeddings add {emb_params:,} dedicated LM-head params "
          f"(tie_embeddings={tied}).")

    print("\nTraining the full Qwen2.5 recipe on the copy task:")
    start = time.time()
    trained, curve, acc = train(cfg, args, record=True, seed=0)
    elapsed = time.time() - start
    print(f"\nFull recipe: exact-copy accuracy {acc*100:.1f}% (trained in {elapsed:.1f}s)")

    # Confirm the distinctive features actually LEARNED something.
    qb = trained.layers[0].self_attn.q_proj.bias.detach()
    kb = trained.layers[0].self_attn.k_proj.bias.detach()
    vb = trained.layers[0].self_attn.v_proj.bias.detach()
    qkv_bias_norm = float((qb.norm() + kb.norm() + vb.norm()).item())
    # Untied: how different are the input embedding and output head now?
    emb = F.normalize(trained.embed_tokens.weight.detach(), dim=-1)
    head = F.normalize(trained.lm_head.weight.detach(), dim=-1)
    mean_cos = float((emb * head).sum(-1).mean().item())
    print(f"\nPost-training evidence the features are load-bearing:")
    print(f"  learned QKV bias L2 norm      : {qkv_bias_norm:.3f}  (was 0 at init)")
    print(f"  mean cos(embed_row, head_row) : {mean_cos:.3f}  (=1.0 would mean tied)")

    # Ablation: remove QKV bias and retrain the same tiny model.
    print(f"\nAblation ({args.ablation_steps} steps each, fresh models):")
    _, _, acc_full = train(Qwen25Config(vocab_size=args.vocab_size, dim=args.dim,
                                        n_layers=args.n_layers, qkv_bias=True),
                           argparse.Namespace(**{**vars(args), "steps": args.ablation_steps}),
                           record=False, seed=1)
    _, _, acc_nobias = train(Qwen25Config(vocab_size=args.vocab_size, dim=args.dim,
                                          n_layers=args.n_layers, qkv_bias=False),
                             argparse.Namespace(**{**vars(args), "steps": args.ablation_steps}),
                             record=False, seed=1)
    print(f"  QKV bias ON  : exact-copy acc {acc_full*100:5.1f}%")
    print(f"  QKV bias OFF : exact-copy acc {acc_nobias*100:5.1f}%")
    print("  (both converge on this easy task; the learned bias norm above is the "
          "decisive\n   evidence the feature is active — its paper benefit is length "
          "extrapolation.)")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "paper": "Qwen2.5 Technical Report (Alibaba 2024, arXiv:2412.15115)",
        "num_parameters": n_params,
        "config": {"dim": cfg.dim, "n_layers": cfg.n_layers, "n_heads": cfg.n_heads,
                   "n_kv_heads": cfg.n_kv_heads, "rope_base": cfg.rope_base},
        "features": [{"name": n, "active": bool(ok)} for n, ok in checks],
        "evidence": {"qkv_bias_l2_norm": round(qkv_bias_norm, 3),
                     "mean_embed_head_cosine": round(mean_cos, 3),
                     "untied_head_params": emb_params},
        "loss_curve": curve,
        "final_accuracy": round(acc, 4),
        "ablation": {"qkv_bias_on_acc": round(acc_full, 4),
                     "qkv_bias_off_acc": round(acc_nobias, 4)},
    }
    (DATA_DIR / "qwen25_demo.json").write_text(json.dumps(out, indent=2))
    print("\nWrote data/qwen25_demo.json (feature report + loss curve for the viz).")

    if acc < 0.9:
        raise SystemExit(f"Recipe did not converge ({acc*100:.1f}% < 90%).")
    if not all(ok for _, ok in checks):
        raise SystemExit("A distinctive feature was not active.")
    print("OK: Qwen2.5 recipe learned the task; all distinctive features confirmed active.")


if __name__ == "__main__":
    main()
