"""Generate a tiny structured language-modeling dataset for the Gemma 2 demo.

The task is a **repetition / induction** task: each sequence is a random prefix
followed by an exact copy of that prefix.  Predicting the copied half requires
*global* attention (look back a long way), while predicting local structure
needs only nearby tokens — a natural fit for Gemma 2's alternating local/global
attention.  Because the copy is deterministic, a trained model becomes very
confident, which makes logit soft-capping's effect easy to see.

Run with:  python data/generate_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

DATA_DIR = Path(__file__).resolve().parent

VOCAB_SIZE = 32
HALF = 16                 # prefix length; full sequence is 2*HALF
NUM_SPECIAL = 2           # 0 = PAD, 1 = SEP (separates the two halves)
PAD, SEP = 0, 1


def make_sequence(gen: torch.Generator) -> list[int]:
    prefix = torch.randint(NUM_SPECIAL, VOCAB_SIZE, (HALF,), generator=gen).tolist()
    # [prefix..., SEP, prefix...]  — the model must reproduce the prefix.
    return prefix + [SEP] + prefix


def make_dataset(n: int, seed: int) -> list[list[int]]:
    gen = torch.Generator().manual_seed(seed)
    return [make_sequence(gen) for _ in range(n)]


def main() -> None:
    train = make_dataset(512, seed=0)
    test = make_dataset(64, seed=1)
    payload = {
        "meta": {
            "vocab_size": VOCAB_SIZE,
            "half": HALF,
            "seq_len": 2 * HALF + 1,
            "special_tokens": {"PAD": PAD, "SEP": SEP},
            "task": "repeat prefix after SEP (induction/copy)",
        },
        "train": train,
        "test": test,
    }
    out = DATA_DIR / "repeat_dataset.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {len(train)} train + {len(test)} test sequences -> {out}")
    print(f"example: {train[0]}")


if __name__ == "__main__":
    main()
