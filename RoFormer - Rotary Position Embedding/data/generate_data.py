"""Generate the "recall the token k positions earlier" task.

This is the relative-position task used to compare RoPE against a model with no
positional information. For a random token sequence ``x``, the target at
position ``t`` is simply::

    y[t] = x[t - k]      for t >= k     (earlier positions are ignored)

Solving it requires knowing the *relative* offset between the current position
and the one k steps back — exactly what RoPE provides. Everything is seeded.

Run with:  python data/generate_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent

VOCAB_SIZE = 20
SHIFT_K = 3
SEED = 0


def make_batch(n: int, seq_len: int, rng: np.random.Generator):
    x = rng.integers(0, VOCAB_SIZE, size=(n, seq_len))
    y = np.full_like(x, -100)  # -100 == ignore_index in the loss
    y[:, SHIFT_K:] = x[:, : seq_len - SHIFT_K]
    return x, y


def main() -> None:
    rng = np.random.default_rng(SEED)
    x, y = make_batch(6, 12, rng)
    payload = {
        "meta": {"vocab_size": VOCAB_SIZE, "shift_k": SHIFT_K, "seed": SEED,
                 "rule": "y[t] = x[t-k]"},
        "examples": [{"x": xi.tolist(), "y": yi.tolist()} for xi, yi in zip(x, y)],
    }
    out = DATA_DIR / "sample_task.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {len(payload['examples'])} sample sequences "
          f"(vocab={VOCAB_SIZE}, k={SHIFT_K}) -> {out}")


if __name__ == "__main__":
    main()
