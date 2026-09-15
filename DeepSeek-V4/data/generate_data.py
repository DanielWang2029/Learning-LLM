"""Generate a tiny teacher-student regression dataset.

A fixed random *teacher* MLP defines a smooth nonlinear target y = f(x). The
demo trains a *student* MLP to match it, which is a clean setting for comparing
optimizers: the loss floor is 0 and both optimizers see identical data.

Writes ``data/regression.json`` (inputs X and targets Y).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def generate(n: int = 512, d_in: int = 16, d_hidden: int = 64, d_out: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d_in)).astype(np.float32)

    # A fixed random teacher MLP (numpy) produces the targets.
    def lin(a, w, b):
        return a @ w + b

    def gelu(z):
        return 0.5 * z * (1.0 + np.tanh(0.79788456 * (z + 0.044715 * z ** 3)))

    w1 = rng.standard_normal((d_in, d_hidden)).astype(np.float32) / np.sqrt(d_in)
    b1 = np.zeros(d_hidden, np.float32)
    w2 = rng.standard_normal((d_hidden, d_out)).astype(np.float32) / np.sqrt(d_hidden)
    b2 = np.zeros(d_out, np.float32)
    Y = lin(gelu(lin(X, w1, b1)), w2, b2).astype(np.float32)
    return X, Y


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    X, Y = generate()
    payload = {"X": X.tolist(), "Y": Y.tolist(),
               "d_in": X.shape[1], "d_out": Y.shape[1]}
    path = out_dir / "regression.json"
    path.write_text(json.dumps(payload))
    print(f"Wrote {path} : n={X.shape[0]} d_in={X.shape[1]} d_out={Y.shape[1]}")


if __name__ == "__main__":
    main()
