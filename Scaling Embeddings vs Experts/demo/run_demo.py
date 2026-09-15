"""End-to-end CPU demo: at a FIXED parameter budget, embeddings vs experts.

We build two tiny language models with the *same backbone* and (approximately)
the *same total parameter count*, differing only in where the extra capacity
goes:

  * embeddings  -> N-gram Embedding tables (larger input capacity).
  * experts     -> a sparse MoE FFN with many experts.

Both train on the same bigram-grammar task for the same number of steps. The
embedding-scaled model reaches lower loss / higher accuracy — the paper's claim
that scaling embeddings can outperform scaling experts at a fixed budget.

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

from src import TinyLM, count_params  # noqa: E402
from src.task import TrigramGrammar  # noqa: E402

DATA_DIR = PAPER_DIR / "data"
SEED = 0

VOCAB = 16
D_MODEL = 24
SEQ_LEN = 24
# capacity knobs, tuned so the two models have ~equal total parameters.
# hash_vocab = VOCAB**3 makes the trigram hash collision-free.
NGRAM_ORDERS = (3,)
HASH_VOCAB = 16 ** 3
N_EXPERTS = 32
TOP_K = 2
D_HIDDEN = 64


def evaluate(model, grammar):
    model.eval()
    rng = torch.Generator().manual_seed(4242)
    inp, tgt = grammar.batch(1024, SEQ_LEN, rng)
    with torch.no_grad():
        logits = model(inp)
    mask = tgt != -100
    loss = torch.nn.functional.cross_entropy(
        logits[mask], tgt[mask]
    ).item()
    acc = (logits.argmax(-1)[mask] == tgt[mask]).float().mean().item()
    return loss, acc


def train(model, grammar, steps, lr=3e-3):
    torch.manual_seed(SEED)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = torch.Generator().manual_seed(SEED)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    curve = []
    for step in range(1, steps + 1):
        model.train()
        inp, tgt = grammar.batch(128, SEQ_LEN, rng)
        logits = model(inp)
        loss = loss_fn(logits.reshape(-1, VOCAB), tgt.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % (steps // 10) == 0 or step == 1:
            _, acc = evaluate(model, grammar)
            curve.append({"step": step, "loss": round(loss.item(), 4), "acc": acc})
    return curve


def main() -> None:
    start = time.time()
    print("=" * 70)
    print("Scaling Embeddings vs Experts — fixed parameter budget")
    print("=" * 70)
    grammar = TrigramGrammar(VOCAB, seed=1)
    print(f"Task: trigram grammar over {VOCAB} tokens (next token = T[a, b, c]),"
          f" seq_len={SEQ_LEN}.\n")

    torch.manual_seed(SEED)
    emb_model = TinyLM(VOCAB, D_MODEL, strategy="embeddings", max_len=SEQ_LEN,
                       ngram_orders=NGRAM_ORDERS, hash_vocab=HASH_VOCAB,
                       n_experts=1, top_k=1, d_hidden=D_HIDDEN)
    torch.manual_seed(SEED)
    exp_model = TinyLM(VOCAB, D_MODEL, strategy="experts", max_len=SEQ_LEN,
                       n_experts=N_EXPERTS, top_k=TOP_K, d_hidden=D_HIDDEN)

    p_emb, p_exp = count_params(emb_model), count_params(exp_model)
    # Exact allocation breakdown: the N-gram tables vs the extra experts.
    base_embed = VOCAB * D_MODEL
    emb_extra = count_params(emb_model.embed) - base_embed        # n-gram tables
    expert_extra = count_params(exp_model.ffn) - count_params(emb_model.ffn)
    backbone = p_emb - emb_extra                                  # shared portion
    print("Fixed budget (total parameters, matched within a few %):")
    print(f"  scale-embeddings  {p_emb:>7,}   (N-gram orders {NGRAM_ORDERS}, hash vocab {HASH_VOCAB})")
    print(f"  scale-experts     {p_exp:>7,}   (top-{TOP_K}/{N_EXPERTS} MoE experts)")
    print(f"  budget difference {abs(p_emb-p_exp)/max(p_emb,p_exp)*100:.1f}%\n")

    steps = 800
    print(f"Training both models for {steps} steps on identical data...")
    emb_curve = train(emb_model, grammar, steps)
    exp_curve = train(exp_model, grammar, steps)

    emb_loss, emb_acc = evaluate(emb_model, grammar)
    exp_loss, exp_acc = evaluate(exp_model, grammar)

    print("\nFinal results on held-out sequences:")
    print(f"  {'allocation':<20}{'params':>10}{'loss':>10}{'accuracy':>12}")
    print(f"  {'scale-embeddings':<20}{p_emb:>10,}{emb_loss:>10.4f}{emb_acc*100:>11.1f}%")
    print(f"  {'scale-experts':<20}{p_exp:>10,}{exp_loss:>10.4f}{exp_acc*100:>11.1f}%")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "paper": "Scaling Embeddings Outperforms Scaling Experts in Language Models (2026)",
        "arxiv": "2601.21204",
        "config": {"vocab": VOCAB, "d_model": D_MODEL, "seq_len": SEQ_LEN,
                   "ngram_orders": list(NGRAM_ORDERS), "hash_vocab": HASH_VOCAB,
                   "n_experts": N_EXPERTS, "top_k": TOP_K},
        "params": {"embeddings": p_emb, "experts": p_exp},
        "breakdown": {"backbone": backbone, "embeddings_extra": emb_extra,
                      "experts_extra": expert_extra},
        "final": {
            "embeddings": {"loss": emb_loss, "acc": emb_acc},
            "experts": {"loss": exp_loss, "acc": exp_acc},
        },
        "curves": {"embeddings": emb_curve, "experts": exp_curve},
    }
    (DATA_DIR / "results.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote data/results.json  (elapsed {time.time()-start:.1f}s)")
    print("Open visualization/index.html to explore the fixed-budget comparison.")

    if not (emb_acc > exp_acc + 0.03 and emb_loss < exp_loss):
        raise SystemExit(
            f"Demo did not show embeddings > experts "
            f"(emb acc {emb_acc*100:.1f}% vs exp acc {exp_acc*100:.1f}%)."
        )
    print("\nOK: at a fixed parameter budget, scaling embeddings beats scaling experts.")


if __name__ == "__main__":
    main()
