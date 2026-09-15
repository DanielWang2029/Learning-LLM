"""Generate a tiny synthetic language for the scaling-laws experiment.

We need a dataset with *real, learnable structure* so that giving a model more
capacity actually lowers its loss (otherwise there is no scaling law to see).
A high-order Markov process is perfect: the next token depends on a window of
previous tokens through a fixed but non-trivial rule, so a bigger model can
capture more of that dependence and reach a lower loss.

The process here is a deterministic-but-scrambled order-``k`` chain:

    next_token = f(hash(last k tokens))  with a small amount of label noise,

which has a well-defined (low but non-zero) entropy floor. Everything is seeded
so the corpus is identical on every run.

Run with:  python data/generate_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent

VOCAB_SIZE = 24
ORDER = 3  # next token depends on the previous 3 tokens
NOISE = 0.10  # fraction of tokens drawn at random -> irreducible entropy floor
N_TOKENS = 24000
BLOCK_SIZE = 32
SEED = 0


def _rule(context: tuple[int, ...], table: np.ndarray) -> int:
    """Map an order-k context to a next token via a fixed random table."""
    idx = 0
    for tok in context:
        idx = (idx * VOCAB_SIZE + tok) % table.shape[0]
    return int(table[idx])


def generate() -> dict:
    rng = np.random.default_rng(SEED)
    # A fixed lookup table gives the deterministic part of the process.
    table = rng.integers(0, VOCAB_SIZE, size=(4096,))

    tokens = list(rng.integers(0, VOCAB_SIZE, size=ORDER).tolist())
    for _ in range(N_TOKENS - ORDER):
        ctx = tuple(tokens[-ORDER:])
        if rng.random() < NOISE:
            nxt = int(rng.integers(0, VOCAB_SIZE))  # noise -> entropy floor
        else:
            nxt = _rule(ctx, table)
        tokens.append(nxt)

    return {
        "meta": {
            "vocab_size": VOCAB_SIZE,
            "order": ORDER,
            "noise": NOISE,
            "block_size": BLOCK_SIZE,
            "n_tokens": len(tokens),
            "seed": SEED,
        },
        "tokens": tokens,
    }


def main() -> None:
    payload = generate()
    out = DATA_DIR / "corpus.json"
    out.write_text(json.dumps(payload))
    meta = payload["meta"]
    print(f"wrote {meta['n_tokens']} tokens (vocab={meta['vocab_size']}, "
          f"order={meta['order']}, noise={meta['noise']}) -> {out}")


if __name__ == "__main__":
    main()
