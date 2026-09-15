"""End-to-end Gemini 1.5 demo (CPU, well under a minute).

Reproduces Gemini 1.5's two headline documented ideas at tiny scale:

1. SPARSE MIXTURE-OF-EXPERTS. The tiny Transformer's feed-forward layers are
   top-k MoE layers. We report the per-expert load to show the router spreads
   tokens across experts (thanks to the load-balancing loss) rather than
   collapsing onto one.

2. LONG-CONTEXT NEAR-PERFECT RECALL. The model is trained on a
   needle-in-a-haystack associative-recall task: read many key->value pairs, then
   answer a query about one planted key. We then measure recall across a grid of
   (needle position × context length) — including contexts LONGER than any seen
   in training — and show it stays high everywhere.

Writes ``data/gemini_results.json`` (recall heatmap + expert loads) for the viz.

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

from src import GeminiConfig, GeminiMoE, VOCAB_SIZE, make_batch, make_eval_batch  # noqa: E402

DATA_DIR = PAPER_DIR / "data"


@torch.no_grad()
def recall_at(model, n_fill, needle_frac, bs, gen):
    tokens, target = make_eval_batch(bs, n_fill, needle_frac, gen)
    logits, _, _ = model(tokens)
    pred = logits[:, -1, :].argmax(-1)
    return (pred == target).float().mean().item()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--min-pairs", type=int, default=4)
    p.add_argument("--max-train-pairs", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--n-experts", type=int, default=4)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--eval-bs", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    cfg = GeminiConfig(vocab_size=VOCAB_SIZE, dim=64, n_layers=2, n_heads=4,
                       moe_hidden=128, n_experts=args.n_experts, top_k=args.top_k)
    model = GeminiMoE(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    n_params = sum(pm.numel() for pm in model.parameters())

    print("=" * 74)
    print("Gemini 1.5 demo — arXiv:2403.05530 (sparse MoE + long-context recall)")
    print("=" * 74)
    print(f"params={n_params:,} | MoE: {cfg.n_experts} experts, top-{cfg.top_k} routing "
          f"| trained on haystacks of {args.min_pairs}-{args.max_train_pairs} distractors\n")

    start = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        n_fill = int(torch.randint(args.min_pairs, args.max_train_pairs + 1, (1,), generator=gen))
        tokens, target = make_batch(args.batch_size, n_fill, gen)
        logits, aux, _ = model(tokens)
        loss = F.cross_entropy(logits[:, -1, :], target) + aux
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == 1:
            r = recall_at(model, args.max_train_pairs, 0.5, args.eval_bs, gen)
            print(f"  step {step:4d}/{args.steps} | loss {loss.item():.4f} | "
                  f"recall@{args.max_train_pairs}-haystack(mid) {r*100:5.1f}%")
    elapsed = time.time() - start

    # ---- Needle-in-a-haystack recall heatmap: position x context length ----
    lengths = [6, 12, 20, 28, 36, 44]   # 36, 44 are LONGER than trained (extrapolation)
    pos_fracs = [0.0, 0.25, 0.5, 0.75, 1.0]
    print(f"\nNeedle-in-a-haystack recall (%) — rows = needle depth, cols = haystack length:")
    header = "         " + "".join(f"{L:>7}" for L in lengths)
    print(header)
    heatmap = []
    for pf in pos_fracs:
        row = [recall_at(model, L, pf, args.eval_bs, gen) for L in lengths]
        heatmap.append(row)
        label = f"pos {int(pf*100):>3}%"
        print(f"  {label} " + "".join(f"{v*100:6.1f}%" for v in row))

    overall = sum(sum(r) for r in heatmap) / (len(heatmap) * len(lengths))
    trained_cols = [i for i, L in enumerate(lengths) if L <= args.max_train_pairs]
    trained_recall = (sum(heatmap[r][c] for r in range(len(heatmap)) for c in trained_cols)
                      / (len(heatmap) * len(trained_cols)))
    print(f"\nOverall recall: {overall*100:.1f}%  |  within trained lengths: {trained_recall*100:.1f}%")

    # ---- MoE expert load (averaged over a batch) ----
    with torch.no_grad():
        tokens, _ = make_batch(args.eval_bs, args.max_train_pairs, gen)
        _, _, loads = model(tokens)
    loads_avg = torch.stack(loads).mean(0)  # average across layers
    # top-k routing sends each token to k experts, so per-expert load sums to k.
    ideal = args.top_k / cfg.n_experts
    print(f"\nMoE expert load (fraction of tokens routed to each; ideal = {ideal:.3f} each):")
    print("  " + "  ".join(f"E{i}:{loads_avg[i]:.3f}" for i in range(cfg.n_experts)))

    print(f"\nTrained in {elapsed:.1f}s on CPU.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "paper": "Gemini 1.5 (Google 2024, arXiv:2403.05530)",
        "config": {"params": n_params, "n_experts": cfg.n_experts, "top_k": cfg.top_k,
                   "trained_pairs": [args.min_pairs, args.max_train_pairs],
                   "seconds": round(elapsed, 1)},
        "heatmap": {"lengths": lengths, "pos_fracs": pos_fracs,
                    "recall": [[round(v, 4) for v in row] for row in heatmap]},
        "overall_recall": round(overall, 4),
        "trained_recall": round(trained_recall, 4),
        "expert_load": [round(float(x), 4) for x in loads_avg],
        "ideal_load": round(ideal, 4),
    }
    (DATA_DIR / "gemini_results.json").write_text(json.dumps(out, indent=2))
    print("Wrote data/gemini_results.json (recall heatmap + expert loads for the viz).")

    if trained_recall < 0.9:
        raise SystemExit(f"Recall within trained lengths too low ({trained_recall*100:.1f}%).")
    max_load = float(loads_avg.max())
    if max_load > 0.6:
        raise SystemExit(f"MoE router collapsed onto one expert (max load {max_load:.2f}).")
    print("OK: MoE routes across experts AND the model recalls the needle across "
          "positions/lengths.")


if __name__ == "__main__":
    main()
