"""Generate a toy "typed token" task for the Switch MoE demo.

Every token id encodes two things:

    type  g   = token %  N_TYPES        (which of a few input "kinds" it is)
    value v   = token // N_TYPES        (a payload within that kind)

The target label is a *type-specific* function of the value::

    label = TABLE[g][v]

so the network must apply a different transformation depending on the token's
type. That is exactly the situation where a Mixture-of-Experts helps: the router
can send each type to its own expert, letting experts specialize. Types are
uniform, so a well-balanced router uses every expert about equally.

Run with:  python data/generate_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent

N_TYPES = 4
VALUES_PER_TYPE = 8
VOCAB_SIZE = N_TYPES * VALUES_PER_TYPE  # 32
N_CLASSES = 6
SEED = 0


def build_table(rng: np.random.Generator) -> np.ndarray:
    """TABLE[g][v] -> class label, a distinct random map per type."""
    return rng.integers(0, N_CLASSES, size=(N_TYPES, VALUES_PER_TYPE))


def token_to_label(token: np.ndarray, table: np.ndarray) -> np.ndarray:
    g = token % N_TYPES
    v = token // N_TYPES
    return table[g, v]


def config() -> dict:
    return {"n_types": N_TYPES, "values_per_type": VALUES_PER_TYPE,
            "vocab_size": VOCAB_SIZE, "n_classes": N_CLASSES, "seed": SEED}


def main() -> None:
    rng = np.random.default_rng(SEED)
    table = build_table(rng)
    sample = rng.integers(0, VOCAB_SIZE, size=(4, 12))
    payload = {
        "meta": config(),
        "label_table": table.tolist(),
        "examples": [
            {"tokens": s.tolist(),
             "types": (s % N_TYPES).tolist(),
             "labels": token_to_label(s, table).tolist()}
            for s in sample
        ],
    }
    out = DATA_DIR / "typed_task.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote typed-token task (vocab={VOCAB_SIZE}, types={N_TYPES}, "
          f"classes={N_CLASSES}) -> {out}")


if __name__ == "__main__":
    main()
