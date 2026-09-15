"""Generate the synthetic corpus for the Chinchilla compute-optimal demo.

Same idea as the scaling-laws corpus: a high-order Markov process with a small
noise floor gives data with real, learnable structure and a well-defined
entropy floor, so both model size and amount of training genuinely matter.
Everything is seeded so the corpus is identical on every run.

Run with:  python data/generate_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent

VOCAB_SIZE = 24
ORDER = 3
NOISE = 0.10
N_TOKENS = 60000  # larger corpus: the "many steps" runs must not repeat data too fast
BLOCK_SIZE = 32
SEED = 0


def _rule(context, table):
    idx = 0
    for tok in context:
        idx = (idx * VOCAB_SIZE + tok) % table.shape[0]
    return int(table[idx])


def generate() -> dict:
    rng = np.random.default_rng(SEED)
    table = rng.integers(0, VOCAB_SIZE, size=(4096,))
    tokens = list(rng.integers(0, VOCAB_SIZE, size=ORDER).tolist())
    for _ in range(N_TOKENS - ORDER):
        ctx = tuple(tokens[-ORDER:])
        if rng.random() < NOISE:
            nxt = int(rng.integers(0, VOCAB_SIZE))
        else:
            nxt = _rule(ctx, table)
        tokens.append(nxt)
    return {
        "meta": {"vocab_size": VOCAB_SIZE, "order": ORDER, "noise": NOISE,
                 "block_size": BLOCK_SIZE, "n_tokens": len(tokens), "seed": SEED},
        "tokens": tokens,
    }


def main() -> None:
    payload = generate()
    out = DATA_DIR / "corpus.json"
    out.write_text(json.dumps(payload))
    m = payload["meta"]
    print(f"wrote {m['n_tokens']} tokens (vocab={m['vocab_size']}, "
          f"order={m['order']}, noise={m['noise']}) -> {out}")


if __name__ == "__main__":
    main()
