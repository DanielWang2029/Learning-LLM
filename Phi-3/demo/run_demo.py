"""Phi-3 "textbooks are all you need" — data quality beats quantity (arXiv 2404.14219).

We train the SAME tiny language model, with the SAME compute budget (same steps,
same batch size, same tokens-per-step), on two datasets:

  * CLEAN : a SMALL, perfectly structured "textbook" corpus.
  * NOISY : a LARGER "web-scraped" corpus where 30% of tokens are corrupted.

Both are evaluated on the same held-out CLEAN test set.  Despite seeing more data,
the noisy-trained model generalizes worse — quality, not quantity, is what counts.

Run with:  python demo/run_demo.py       (well under a minute on CPU)
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
torch.set_num_threads(1)  # shared box: avoid thread oversubscription

from src.data import (  # noqa: E402
    make_clean_dataset,
    make_noisy_dataset,
    next_token_accuracy,
    VOCAB_SIZE,
    SEQ_LEN,
)
from src.model import GPTConfig, TinyGPT  # noqa: E402

DATA_DIR = ROOT / "data"
SEED = 0
STEPS = 500
BATCH = 64
LR = 3e-3
CLEAN_N = 256          # small, curated
NOISY_N = 4096         # 16x larger, but systematically corrupted


def banner(t: str) -> None:
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def train(dataset: torch.Tensor, test: torch.Tensor, label: str):
    """Train a fresh TinyGPT for STEPS steps; return model, accuracy curve."""
    cfg = GPTConfig(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN - 1)
    torch.manual_seed(SEED)
    model = TinyGPT(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(SEED)
    n = dataset.size(0)

    curve = {"step": [], "acc": []}
    tokens_seen = 0
    for step in range(1, STEPS + 1):
        model.train()
        idx = torch.randint(0, n, (BATCH,), generator=gen)
        batch = dataset[idx]
        x, y = batch[:, :-1], batch[:, 1:]
        logits = model(x)
        loss = loss_fn(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        tokens_seen += x.numel()

        if step % 25 == 0 or step == 1:
            acc = next_token_accuracy(model, test)
            curve["step"].append(step)
            curve["acc"].append(round(acc, 4))
            if step % 100 == 0 or step == 1:
                print(f"  [{label:>5}] step {step:3d} | loss {loss.item():.3f} "
                      f"| clean-test acc {acc*100:5.1f}%")
    return model, curve, tokens_seen


def main() -> None:
    t0 = time.time()
    print("Phi-3 in miniature — data QUALITY vs QUANTITY at equal compute")

    clean_train = make_clean_dataset(CLEAN_N, seed=1)
    noisy_train = make_noisy_dataset(NOISY_N, seed=2)
    test = make_clean_dataset(512, seed=999)     # held-out, CLEAN

    cfg = GPTConfig(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN - 1)
    params = TinyGPT(cfg).num_params()

    banner("SETUP — identical model & compute, different data")
    print(f"  model params        : {params:,}  (same for both runs)")
    print(f"  training steps      : {STEPS}   batch size: {BATCH}   "
          f"(same compute for both)")
    print(f"  CLEAN dataset       : {CLEAN_N:>5} sequences, 0% corrupted "
          f"(structured 'textbook')")
    print(f"  NOISY dataset       : {NOISY_N:>5} sequences, ~40% of contexts "
          f"systematically WRONG ('web scrape', {NOISY_N//CLEAN_N}x MORE data)")
    print(f"  rule (both clean seqs): next = (prev1 + prev2) mod {VOCAB_SIZE}")

    banner("TRAIN on CLEAN 'textbook' data")
    clean_model, clean_curve, clean_tokens = train(clean_train, test, "clean")

    banner("TRAIN on NOISY 'web' data (larger, corrupted)")
    noisy_model, noisy_curve, noisy_tokens = train(noisy_train, test, "noisy")

    clean_acc = next_token_accuracy(clean_model, test)
    noisy_acc = next_token_accuracy(noisy_model, test)

    banner("RESULTS — evaluated on the SAME held-out CLEAN test set")
    print(f"  CLEAN-trained model : {clean_acc*100:5.1f}%  "
          f"(saw {CLEAN_N} clean seqs, {clean_tokens:,} tokens)")
    print(f"  NOISY-trained model : {noisy_acc*100:5.1f}%  "
          f"(saw {NOISY_N} seqs, {noisy_tokens:,} tokens — MORE data)")
    print(f"  quality advantage   : {(clean_acc-noisy_acc)*100:+.1f} points "
          f"for the curated data, at equal compute")

    # Example clean vs noisy sequences for the visualization.
    from src.data import rule_reconstruct
    ex_clean = make_clean_dataset(3, seed=100).tolist()
    ex_noisy = make_noisy_dataset(3, seed=100).tolist()
    # Which positions carry the systematic error (differ from the pure rule).
    noisy_flags = [[int(seq[i] != rule_reconstruct(seq)[i]) for i in range(len(seq))]
                   for seq in ex_noisy]

    DATA_DIR.mkdir(exist_ok=True)
    payload = {
        "config": {
            "params": params, "steps": STEPS, "batch": BATCH,
            "vocab_size": VOCAB_SIZE, "seq_len": SEQ_LEN,
            "clean_n": CLEAN_N, "noisy_n": NOISY_N,
            "rule": f"next = (prev1 + prev2) mod {VOCAB_SIZE}",
        },
        "results": {
            "clean_acc": clean_acc, "noisy_acc": noisy_acc,
            "clean_tokens": clean_tokens, "noisy_tokens": noisy_tokens,
        },
        "curves": {"clean": clean_curve, "noisy": noisy_curve},
        "examples": {
            "clean": ex_clean,
            "noisy": ex_noisy,
            "noisy_corrupt_flags": noisy_flags,
        },
    }
    out = DATA_DIR / "phi3_results.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n  wrote {out}")
    print(f"  total time: {time.time()-t0:.1f}s")

    # --- Sanity asserts ------------------------------------------------------
    assert clean_acc > 0.9, f"clean model should learn the rule ({clean_acc:.2f})"
    assert clean_acc > noisy_acc + 0.15, (
        f"curated data should clearly win (clean {clean_acc:.2f} vs "
        f"noisy {noisy_acc:.2f})")
    assert noisy_tokens > clean_tokens or NOISY_N > CLEAN_N, "noisy set is larger"
    print("\nOK: at equal compute, the small CLEAN dataset beats the larger NOISY "
          "one — data quality > quantity.")


if __name__ == "__main__":
    main()
