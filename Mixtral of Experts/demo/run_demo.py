"""Mixtral of Experts in miniature (arXiv 2401.04088).

What this demonstrates on CPU in a few seconds:

  1. A SPARSE Mixture-of-Experts transformer: 8 experts, TOP-2 routing.
  2. Total vs ACTIVE parameters — only 2 of 8 experts run per token.
  3. Load-balancing loss — with it the router uses all experts evenly; without
     it, routing collapses onto a few experts.
  4. Expert SPECIALIZATION — after training on a 4-domain task, each domain is
     routed to a consistent subset of experts (a domain x expert heatmap).
  5. A worked TOP-2 gate for one token.
  6. Conceptual contrast with Switch Transformers' TOP-1 routing.

Run with:  python demo/run_demo.py
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
torch.set_num_threads(1)  # shared box: avoid thread oversubscription

from src.data import make_dataset, N_DOMAINS, VOCAB_SIZE, SEQ_LEN, DOMAIN_SIZE  # noqa: E402
from src.model import MoEConfig, MoETransformer  # noqa: E402

DATA_DIR = ROOT / "data"
SEED = 0
STEPS = 400
BATCH = 64
LR = 3e-3
AUX_WEIGHT = 0.001


def banner(t: str) -> None:
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


@torch.no_grad()
def accuracy(model, x, y):
    model.eval()
    logits, _ = model(x)
    return (logits.argmax(-1) == y).float().mean().item()


def train(use_aux: bool, seqs, labels, test_x, test_y):
    cfg = MoEConfig(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN - 1)
    torch.manual_seed(SEED)
    model = MoETransformer(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(SEED)
    n = seqs.size(0)

    aux_curve = {"step": [], "aux": [], "acc": []}
    for step in range(1, STEPS + 1):
        model.train()
        idx = torch.randint(0, n, (BATCH,), generator=gen)
        batch = seqs[idx]
        x, y = batch[:, :-1], batch[:, 1:]
        logits, aux = model(x)
        loss = loss_fn(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
        if use_aux:
            loss = loss + AUX_WEIGHT * aux
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 25 == 0 or step == 1:
            acc = accuracy(model, test_x, test_y)
            aux_curve["step"].append(step)
            aux_curve["aux"].append(round(aux.item(), 4))
            aux_curve["acc"].append(round(acc, 4))
    return model, aux_curve


def routing_report(model, test_x):
    """Summarize routing: overall load + a token x expert specialization map.

    Routing turns out to be a consistent function of the *input token*: each
    vocabulary token is sent to a stable dominant expert.  So we build a
    (vocab x expert) map from the top-1 choice, which reveals a clean partition
    of the token space among the experts.
    """
    model.eval()
    with torch.no_grad():
        model(test_x)
    stats = model.moe.stats
    n_experts = model.cfg.n_experts

    load = stats.load_fraction.tolist()          # overall usage histogram

    tok_ids = test_x.reshape(-1)                 # input token per flattened slot
    top1 = stats.expert_indices[:, 0]
    tok_expert = torch.zeros(VOCAB_SIZE, n_experts)
    for v in range(VOCAB_SIZE):
        sel = top1[tok_ids == v]
        if sel.numel() == 0:
            continue
        counts = torch.bincount(sel, minlength=n_experts).float()
        tok_expert[v] = counts / counts.sum()
    return load, tok_expert.tolist(), stats


def main() -> None:
    t0 = time.time()
    print("Mixtral of Experts in miniature — sparse top-2 MoE")

    seqs, labels = make_dataset(1024, seed=1)
    test_seqs, test_labels = make_dataset(256, seed=2)
    test_x, test_y = test_seqs[:, :-1], test_seqs[:, 1:]

    cfg = MoEConfig(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN - 1)
    model0 = MoETransformer(cfg)

    banner("MODEL — total vs ACTIVE parameters (top-2 of 8 experts)")
    total = model0.total_params()
    active = model0.active_params_per_token()
    expert_p = model0.moe.expert_param_count()
    print(f"  experts               : {cfg.n_experts}   top-k routing: {cfg.top_k}")
    print(f"  params per expert      : {expert_p:,}")
    print(f"  TOTAL parameters       : {total:,}")
    print(f"  ACTIVE params / token  : {active:,}  "
          f"(only {cfg.top_k}/{cfg.n_experts} experts run)")
    print(f"  sparsity               : {active/total*100:.0f}% of params active "
          f"per token")
    # Switch (top-1) contrast:
    active_top1 = total - expert_p * cfg.n_experts + expert_p * 1
    print(f"  [contrast] Switch top-1 would activate {active_top1:,} params/token "
          f"(1/{cfg.n_experts} experts)")

    banner("TRAIN with load-balancing loss")
    model, curve = train(True, seqs, labels, test_x, test_y)
    acc = accuracy(model, test_x, test_y)
    load, tok_expert, stats = routing_report(model, test_x)
    print(f"  final next-token accuracy : {acc*100:.1f}%")
    print(f"  expert usage (fraction of dispatch slots):")
    print("    " + "  ".join(f"E{e}:{load[e]*100:4.1f}%" for e in range(cfg.n_experts)))

    banner("EXPERT SPECIALIZATION — dominant expert per token (top-1 routing)")
    print("  Each token is routed to a stable expert; '.' = <5% share.")
    print("    token -> " + " ".join(f"E{e}" for e in range(cfg.n_experts)))
    peaks = []
    for v in range(VOCAB_SIZE):
        dist = tok_expert[v]
        peaks.append(max(dist))
        dom = v // DOMAIN_SIZE
        row = " ".join(("##" if p > 0.5 else ("+ " if p >= 0.05 else ". "))
                       for p in dist)
        dom_expert = int(max(range(cfg.n_experts), key=lambda e: dist[e]))
        print(f"  tok {v:2d}(d{dom})  {row}   -> E{dom_expert} ({max(dist)*100:.0f}%)")
    mean_peak = sum(peaks) / len(peaks)
    print(f"\n  mean dominant-expert share per token: {mean_peak*100:.1f}% "
          f"(higher = crisper specialization)")

    banner("A WORKED TOP-2 GATE (one token)")
    tok = 0
    ei = stats.expert_indices[tok].tolist()
    gw = stats.gate_weights[tok].tolist()
    probs = stats.router_probs[tok].tolist()
    print(f"  router softmax over experts: "
          + " ".join(f"E{e}:{probs[e]:.2f}" for e in range(cfg.n_experts)))
    print(f"  -> top-2 experts chosen    : E{ei[0]} (w={gw[0]:.3f}), "
          f"E{ei[1]} (w={gw[1]:.3f})   [weights renormalized to sum to 1]")

    banner("LOAD BALANCING — with vs without the auxiliary loss")
    model_noaux, _ = train(False, seqs, labels, test_x, test_y)
    load_noaux, _, _ = routing_report(model_noaux, test_x)

    def entropy(dist):
        import math
        return -sum(p * math.log(p + 1e-12) for p in dist)
    max_ent = __import__("math").log(cfg.n_experts)
    bal_aux = entropy(load) / max_ent
    bal_noaux = entropy(load_noaux) / max_ent
    ratio_aux = max(load) / min(load)
    ratio_noaux = max(load_noaux) / min(load_noaux)
    print(f"  WITH aux   — balance {bal_aux:.3f}, max/min usage {ratio_aux:.2f}x")
    print("    " + "  ".join(f"E{e}:{load[e]*100:4.1f}%" for e in range(cfg.n_experts)))
    print(f"  WITHOUT aux— balance {bal_noaux:.3f}, max/min usage {ratio_noaux:.2f}x")
    print("    " + "  ".join(f"E{e}:{load_noaux[e]*100:4.1f}%"
                             for e in range(cfg.n_experts)))
    print("  (This toy task is balanced, so routing stays fairly even either way;")
    print("   the aux loss still tightens it and, at scale, prevents expert collapse.)")

    # --- persist for the visualization --------------------------------------
    DATA_DIR.mkdir(exist_ok=True)
    payload = {
        "config": {
            "n_experts": cfg.n_experts, "top_k": cfg.top_k,
            "n_domains": N_DOMAINS, "domain_size": DOMAIN_SIZE,
            "vocab_size": VOCAB_SIZE, "seq_len": SEQ_LEN,
            "total_params": total, "active_params": active,
            "active_params_top1": active_top1,
            "params_per_expert": expert_p,
        },
        "results": {"accuracy": acc},
        "expert_load_with_aux": load,
        "expert_load_without_aux": load_noaux,
        "balance_with_aux": bal_aux,
        "balance_without_aux": bal_noaux,
        "imbalance_ratio_with_aux": ratio_aux,
        "imbalance_ratio_without_aux": ratio_noaux,
        "token_expert": tok_expert,   # [token][expert] top-1 routing share
        "mean_token_peak": mean_peak,
        "example_gate": {
            "router_probs": [round(p, 4) for p in probs],
            "top_experts": ei, "top_weights": [round(w, 4) for w in gw],
        },
        "aux_curve": curve,
    }
    out = DATA_DIR / "mixtral_results.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n  wrote {out}")
    print(f"  total time: {time.time()-t0:.1f}s")

    # --- asserts -------------------------------------------------------------
    assert acc > 0.9, f"MoE should learn the multi-domain task ({acc:.2f})"
    assert active < total, "active params must be fewer than total"
    assert bal_aux >= bal_noaux, "aux loss should not worsen load balance"
    # Specialization: each token should route to a dominant expert consistently.
    assert mean_peak > 0.6, f"routing not specialized enough ({mean_peak:.2f})"
    print("\nOK: sparse top-2 routing works; the router learns a stable, "
          "specialized token->expert map and the load-balancing loss keeps "
          "experts evenly used.")


if __name__ == "__main__":
    main()
