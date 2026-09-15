"""NSA demo: match full-attention quality while attending to far fewer positions.

We train two identical tiny decoder-only models on a long-range **repeat-
induction** task — one with **Native Sparse Attention** (compression + selection
+ sliding window, gated), one with **full** causal attention — for the same
budget, then compare:

* held-out next-token accuracy on the second half (NSA should match full), and
* the average number of key positions each query attends to (NSA << full), and
* the learned three-branch gate weights.

Repeat-induction: a random block ``base`` of length ``PERIOD`` is repeated twice,
so ``seq = base ++ base``. Predicting a token in the *second* copy requires
attending back one whole period (distance ``PERIOD``) to the matching token in
the first copy — a long-range dependency the local sliding window alone cannot
cover, which forces the selection branch to do real work.

Writes ``data/nsa_run.json`` for the visualization.

Run with:  python demo/run_demo.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import RecallModel  # noqa: E402

SEED = 0
PERIOD = 32            # a random block of this length is repeated twice
NUM_SYMBOLS = 16
VOCAB = NUM_SYMBOLS
SEQ_LEN = 2 * PERIOD   # sequence = base ++ base
BLOCK = 8
N_SELECT = 3
WINDOW = 8             # < PERIOD, so the long-range match needs the SELECTION branch
# Full attention converges almost immediately; NSA's sparse selection needs more
# updates to learn *which* far block holds the induction match, so we give each
# the budget it needs (both remain tiny and CPU-friendly).
STEPS_FULL = 150
STEPS_NSA = 1000
BATCH = 40
# Supervise next-token prediction only where induction is required: predicting a
# token whose index lies in the second copy (target index >= PERIOD). With
# inputs = seq[:, :-1], targets = seq[:, 1:], that is input positions >= PERIOD-1.
PRED_FROM = PERIOD - 1


def make_batch(batch, gen: torch.Generator):
    """Repeat-induction: seq = base ++ base."""
    base = torch.randint(0, NUM_SYMBOLS, (batch, PERIOD), generator=gen)
    return torch.cat([base, base], dim=1)


def train(attn, steps, gen):
    torch.manual_seed(SEED)
    model = RecallModel(
        VOCAB, d_model=64, num_layers=2, num_heads=2, d_ff=128, max_len=SEQ_LEN + 2,
        attn=attn, block=BLOCK, n_select=N_SELECT, window=WINDOW,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    for step in range(1, steps + 1):
        model.train()
        seq = make_batch(BATCH, gen)
        inp, tgt = seq[:, :-1], seq[:, 1:]
        logits = model(inp)
        # dense supervision over the second-copy positions only
        loss = F.cross_entropy(
            logits[:, PRED_FROM:, :].reshape(-1, VOCAB),
            tgt[:, PRED_FROM:].reshape(-1),
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 150 == 0 or step == 1:
            print(f"  [{attn}] step {step:4d}/{steps} | loss {loss.item():.4f}")
    return model


@torch.no_grad()
def evaluate(model, gen, n=1000, collect=False):
    model.eval()
    seq = make_batch(n, gen)
    inp, tgt = seq[:, :-1], seq[:, 1:]
    logits = model(inp, collect=collect)
    pred = logits[:, PRED_FROM:, :].argmax(dim=-1)
    acc = (pred == tgt[:, PRED_FROM:]).float().mean().item()
    return acc


def main() -> None:
    torch.manual_seed(SEED)
    print("Native Sparse Attention vs full attention — repeat-induction")
    print(f"seq_len={SEQ_LEN} period={PERIOD} | NSA: block={BLOCK} select={N_SELECT} window={WINDOW}\n")

    start = time.time()
    print("Training FULL-attention model ...")
    m_full = train("full", STEPS_FULL, torch.Generator().manual_seed(SEED))
    print("Training NSA model ...")
    m_nsa = train("nsa", STEPS_NSA, torch.Generator().manual_seed(SEED))
    print(f"trained both in {time.time() - start:.1f}s\n")

    acc_full = evaluate(m_full, torch.Generator().manual_seed(SEED + 1))
    acc_nsa = evaluate(m_nsa, torch.Generator().manual_seed(SEED + 1), collect=True)
    stats = m_nsa.attn_stats()

    # Positions touched by a *full-context* query (the last position sees the
    # whole sequence): full attends to all of them, NSA only to its sparse budget.
    nsa_pos = stats["nsa_positions_last"]
    full_pos = stats["full_positions_last"]
    saved = (1 - nsa_pos / full_pos) * 100

    print("Held-out second-half next-token accuracy:")
    print(f"  full attention: {acc_full * 100:5.1f}%")
    print(f"  NSA           : {acc_nsa * 100:5.1f}%\n")
    print(f"Key positions touched by a full-context query (context = {SEQ_LEN - 1}):")
    print(f"  full attention: {full_pos:5.1f}")
    print(f"  NSA           : {nsa_pos:5.1f}   ({saved:.0f}% fewer positions)\n")
    print("Learned NSA gate weights (avg over queries):")
    print(f"  compression: {stats['gate_cmp']:.2f} | selection: {stats['gate_slc']:.2f} | window: {stats['gate_win']:.2f}")

    # Analytical scaling: NSA's per-query budget = compressed coarse tokens
    # (ceil(L/block)) + selected fine (n_select*block) + sliding window, while
    # full attention grows linearly with context length L. This is where NSA's
    # savings become dramatic (the paper operates at 64k+ tokens).
    scaling = []
    for L in [64, 128, 256, 512, 1024, 4096, 16384, 65536]:
        coarse = -(-L // BLOCK)  # ceil
        nsa_cost = min(L, coarse + N_SELECT * BLOCK + WINDOW)
        scaling.append({"length": L, "full": L, "nsa": nsa_cost})

    out = {
        "meta": {
            "seq_len": SEQ_LEN, "period": PERIOD, "block": BLOCK,
            "n_select": N_SELECT, "window": WINDOW, "num_symbols": NUM_SYMBOLS,
        },
        "accuracy": {"full": acc_full, "nsa": acc_nsa},
        "positions": {"full": full_pos, "nsa": nsa_pos, "saved_pct": saved,
                       "context": SEQ_LEN - 1},
        "gate": {"compression": stats["gate_cmp"], "selection": stats["gate_slc"], "window": stats["gate_win"]},
        "scaling": scaling,
    }
    out_path = ROOT / "data" / "nsa_run.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")

    assert acc_nsa > 0.85, f"NSA accuracy too low ({acc_nsa:.2f})"
    assert acc_nsa > acc_full - 0.1, "NSA should roughly match full attention"
    assert nsa_pos < full_pos, "NSA should touch fewer positions than full attention"
    print("\nOK: NSA matches full-attention quality while touching fewer positions.")


if __name__ == "__main__":
    main()
