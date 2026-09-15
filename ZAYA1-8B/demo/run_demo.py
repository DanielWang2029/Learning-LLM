"""End-to-end CPU demo for ZAYA1-8B's Compressed Convolutional Attention (CCA).

We train three models on a content-based retrieval task (find the single marked
token, return its payload):

  * full attention        the baseline (KV-cache = L positions).
  * CCA, compress = 4      conv-downsample K/V by 4  (KV-cache = L/4).
  * CCA, compress = 8      conv-downsample K/V by 8  (KV-cache = L/8, the paper's
                           reported KV compression rate).

CCA matches full attention's accuracy while shrinking the attention matrix and
the KV-cache by the compression factor. The demo reports accuracy, KV-cache
size, attention-matrix size, and how often CCA agrees with the full model.

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
    FullAttention,
    CompressedConvAttention,
    Classifier,
    kv_cache_size,
)
from src.task import MarkedRetrieval  # noqa: E402

DATA_DIR = PAPER_DIR / "data"
SEED = 0

VOCAB = 16
SEQ_LEN = 48
D_MODEL = 48
N_HEADS = 4


def train(model, task, steps=500, lr=3e-3):
    torch.manual_seed(SEED)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    rng = torch.Generator().manual_seed(SEED)
    for _ in range(steps):
        model.train()
        t, m, y = task.batch(256, rng)
        logits = model(t, m)
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()


@torch.no_grad()
def predict(model, task, seed=9, batch=4000):
    model.eval()
    rng = torch.Generator().manual_seed(seed)
    t, m, y = task.batch(batch, rng)
    pred = model(t, m).argmax(-1)
    return pred, y


def attention_sample(model, task):
    """Return an averaged attention map (over heads) for one example."""
    model.eval()
    rng = torch.Generator().manual_seed(123)
    t, m, y = task.batch(1, rng)
    with torch.no_grad():
        model(t, m)
    attn = model.attn.last_attn[0].mean(dim=0)   # (L, Lc) averaged over heads
    marked = int(m[0].argmax())
    return attn.tolist(), marked, int(y[0])


def main() -> None:
    start = time.time()
    print("=" * 70)
    print("ZAYA1-8B — Compressed Convolutional Attention (CCA)")
    print("=" * 70)
    print(f"Task: retrieve the marked token's payload. seq_len={SEQ_LEN}, "
          f"vocab={VOCAB}.\n")
    task = MarkedRetrieval(VOCAB, SEQ_LEN)

    configs = [("full", None), ("cca-4", 4), ("cca-8", 8)]
    models = {}
    for name, r in configs:
        torch.manual_seed(SEED)
        attn = (FullAttention(D_MODEL, N_HEADS) if r is None
                else CompressedConvAttention(D_MODEL, N_HEADS, compress=r))
        models[name] = Classifier(VOCAB, VOCAB, attn, D_MODEL)

    print("Training full attention + CCA (compress 4, 8)...")
    for name, _ in configs:
        train(models[name], task)

    full_pred, y = predict(models["full"], task)
    results = {}
    print("\nResults on held-out sequences:")
    print(f"  {'model':<10}{'accuracy':>10}{'KV positions':>14}{'KV scalars':>13}"
          f"{'attn entries':>14}{'agree w/ full':>15}")
    for name, r in configs:
        pred, y = predict(models[name], task)
        acc = (pred == y).float().mean().item()
        agree = (pred == full_pred).float().mean().item()
        comp = r or 1
        kv_pos = models[name].attn.kv_positions(SEQ_LEN)
        kv_scalars = kv_cache_size(SEQ_LEN, D_MODEL, comp)
        attn_entries = SEQ_LEN * kv_pos
        results[name] = {
            "accuracy": acc, "compress": comp, "kv_positions": kv_pos,
            "kv_scalars": kv_scalars, "attn_entries": attn_entries,
            "agree_with_full": agree,
        }
        print(f"  {name:<10}{acc*100:>9.1f}%{kv_pos:>14}{kv_scalars:>13}"
              f"{attn_entries:>14}{agree*100:>14.1f}%")

    kv_full = results["full"]["kv_scalars"]
    print(f"\nKV-cache reduction:  cca-4 = {kv_full/results['cca-4']['kv_scalars']:.0f}x,"
          f"  cca-8 = {kv_full/results['cca-8']['kv_scalars']:.0f}x  smaller than full.")

    attn_full, marked_full, label_full = attention_sample(models["full"], task)
    attn_cca, marked_cca, label_cca = attention_sample(models["cca-8"], task)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "paper": "ZAYA1-8B Technical Report (2026)",
        "arxiv": "2605.05365",
        "config": {"vocab": VOCAB, "seq_len": SEQ_LEN, "d_model": D_MODEL,
                   "n_heads": N_HEADS},
        "results": results,
        "attention_full": {"map": attn_full, "marked_pos": marked_full,
                           "label": label_full, "seq_len": SEQ_LEN},
        "attention_cca8": {"map": attn_cca, "marked_pos": marked_cca,
                           "label": label_cca, "compress": 8},
    }
    (DATA_DIR / "results.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote data/results.json  (elapsed {time.time()-start:.1f}s)")
    print("Open visualization/index.html to explore CCA compression + savings.")

    ok = (results["cca-8"]["accuracy"] > 0.9
          and results["cca-8"]["accuracy"] > results["full"]["accuracy"] - 0.05)
    if not ok:
        raise SystemExit(
            f"CCA did not match full attention "
            f"(cca-8 {results['cca-8']['accuracy']*100:.1f}% vs "
            f"full {results['full']['accuracy']*100:.1f}%)."
        )
    print("\nOK: CCA matches full-attention quality while shrinking the KV-cache "
          f"{kv_full/results['cca-8']['kv_scalars']:.0f}x.")


if __name__ == "__main__":
    main()
