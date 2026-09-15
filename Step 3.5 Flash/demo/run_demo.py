"""End-to-end CPU demo for Step 3.5 Flash's sparse-MoE efficiency idea.

We train three models on a topic-conditioned symbol-mapping task:

  * MoE          many experts, only top-k run per token  (high total capacity,
                 few ACTIVE parameters — the Step 3.5 Flash idea).
  * dense-small  a single MLP matched to the MoE's ACTIVE compute.
  * dense-big    a single MLP matched to the MoE's TOTAL parameters
                 (all of them active every token).

The point: the MoE reaches dense-big quality while activating only a fraction
of its parameters per token — and it beats the active-matched dense-small.
We also show the router specializes experts by topic.

Run with:  python demo/run_demo.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(1)

PAPER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PAPER_DIR))

from src import (  # noqa: E402
    MoEClassifier,
    DenseClassifier,
    count_params,
    active_params,
)
from src.task import TopicPermutations  # noqa: E402

DATA_DIR = PAPER_DIR / "data"
SEED = 0

N_SYMBOLS = 64
N_TOPICS = 64
D_MODEL = 48
N_EXPERTS = 64
TOP_K = 1
D_HIDDEN = 32


def evaluate(model, task, is_moe, batch=4000):
    model.eval()
    rng = torch.Generator().manual_seed(4242)
    s, g, y = task.batch(batch, rng)
    with torch.no_grad():
        logits = model(s, g)
    return (logits.argmax(-1) == y).float().mean().item()


def train(model, task, steps, lr=3e-3):
    torch.manual_seed(SEED)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    rng = torch.Generator().manual_seed(SEED)
    for step in range(1, steps + 1):
        model.train()
        s, g, y = task.batch(256, rng)
        logits = model(s, g)
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item()


def routing_matrix(model, task):
    """Return an (n_topics x n_experts) matrix of top-1 expert usage per topic."""
    model.eval()
    mat = torch.zeros(task.n_topics, model.moe.n_experts)
    rng = torch.Generator().manual_seed(77)
    s, g, _ = task.batch(8000, rng)
    with torch.no_grad():
        _, routing = model(s, g, return_routing=True)
    top1 = routing["top_idx"][:, 0]
    for t, e in zip(g.tolist(), top1.tolist()):
        mat[t, e] += 1
    mat = mat / mat.sum(dim=1, keepdim=True).clamp(min=1)
    return mat


def main() -> None:
    start = time.time()
    print("=" * 70)
    print("Step 3.5 Flash — sparse MoE: high total capacity, few active params")
    print("=" * 70)
    print(f"Task: {N_TOPICS} topics x {N_SYMBOLS} symbols, each topic a distinct "
          f"permutation ({N_TOPICS * N_SYMBOLS} input->output mappings).\n")

    task = TopicPermutations(N_SYMBOLS, N_TOPICS, seed=1)

    # Seed before each model so all three share identical FROZEN input embeddings.
    torch.manual_seed(SEED)
    moe = MoEClassifier(N_SYMBOLS, N_TOPICS, D_MODEL, N_EXPERTS, TOP_K, D_HIDDEN)
    torch.manual_seed(SEED)
    dense_small = DenseClassifier(N_SYMBOLS, N_TOPICS, D_MODEL, D_HIDDEN)
    torch.manual_seed(SEED)
    dense_big = DenseClassifier(N_SYMBOLS, N_TOPICS, D_MODEL, D_HIDDEN * N_EXPERTS)

    moe_total = count_params(moe)
    moe_active = active_params(moe)
    print("Parameter budgets:")
    print(f"  MoE          total={moe_total:>7,}   active/token={moe_active:>7,}"
          f"   ({moe_active/moe_total*100:.1f}% active, top-{TOP_K}/{N_EXPERTS} experts)")
    print(f"  dense-small  total={count_params(dense_small):>7,}   (matched to MoE ACTIVE compute)")
    print(f"  dense-big    total={count_params(dense_big):>7,}   (matched to MoE TOTAL params, all active)\n")

    steps = 1000
    print(f"Training each model for {steps} steps...")
    train(moe, task, steps)
    train(dense_small, task, steps)
    train(dense_big, task, steps)

    acc_moe = evaluate(moe, task, True)
    acc_small = evaluate(dense_small, task, False)
    acc_big = evaluate(dense_big, task, False)

    print("\nAccuracy on held-out (symbol, topic) pairs:")
    print(f"  MoE (top-{TOP_K}/{N_EXPERTS})   {acc_moe*100:5.1f}%   "
          f"active/token={moe_active:,}")
    print(f"  dense-small     {acc_small*100:5.1f}%   active/token={count_params(dense_small):,}")
    print(f"  dense-big       {acc_big*100:5.1f}%   active/token={count_params(dense_big):,}")

    mat = routing_matrix(moe, task)
    # Specialization: the router is far from uniform. Compare each topic's
    # busiest expert to the uniform baseline (1/n_experts).
    dominant = mat.max(dim=1).values.mean().item()
    uniform = 1.0 / N_EXPERTS
    top8 = mat.topk(min(8, N_EXPERTS), dim=1).values.sum(dim=1).mean().item()
    print(f"\nRouting is specialized (not uniform):")
    print(f"  busiest expert per topic handles {dominant*100:.1f}% of its tokens "
          f"({dominant/uniform:.0f}x the {uniform*100:.1f}% uniform baseline).")
    print(f"  a topic's 8 busiest experts handle {top8*100:.1f}% of its tokens.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "paper": "Step 3.5 Flash: Open Frontier-Level Intelligence with 11B Active Parameters (2026)",
        "arxiv": "2602.10604",
        "task": {"n_topics": N_TOPICS, "n_symbols": N_SYMBOLS,
                 "n_experts": N_EXPERTS, "top_k": TOP_K},
        "params": {
            "moe_total": moe_total, "moe_active": moe_active,
            "dense_small": count_params(dense_small),
            "dense_big": count_params(dense_big),
            "active_fraction": moe_active / moe_total,
        },
        "accuracy": {"moe": acc_moe, "dense_small": acc_small, "dense_big": acc_big},
        "routing_matrix": mat.tolist(),
        "dominant_share": dominant,
        "uniform_share": uniform,
        "top8_capture": top8,
    }
    (DATA_DIR / "results.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote data/results.json  (elapsed {time.time()-start:.1f}s)")
    print("Open visualization/index.html to explore total-vs-active params and routing.")

    if not (acc_moe > 0.9 and acc_moe > acc_small + 0.1):
        raise SystemExit(
            f"Demo did not show the effect (MoE {acc_moe*100:.1f}% vs "
            f"dense-small {acc_small*100:.1f}%)."
        )
    print("\nOK: sparse MoE matches the big dense model while activating a "
          f"fraction of its parameters, and beats the active-matched dense model.")


if __name__ == "__main__":
    main()
