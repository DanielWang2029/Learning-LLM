"""Delta-rule associative memory (the fast-weight mechanism behind DDL).

A single matrix state ``W ∈ R^{d_v × d_k}`` stores key->value bindings. Reading
is a matrix-vector product ``v_hat = W k``; writing uses the **delta rule**
(a.k.a. Widrow-Hoff / online least-squares update):

    v_hat = W k                       # current read for key k
    W <- W + β (v - v_hat) kᵀ         # write the residual toward target v

With unit-norm keys, β = 1 makes the read of key k exactly equal to v after the
write, and (because it subtracts the current read first) rewriting an existing
key REPLACES its value instead of accumulating a superposition. This is the
building block Deep Delta Learning reuses as a residual interface over depth.
"""

from __future__ import annotations

import torch


def _unit(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


class DeltaAssociativeMemory:
    def __init__(self, d_key: int, d_val: int) -> None:
        self.d_key = d_key
        self.d_val = d_val
        self.W = torch.zeros(d_val, d_key)

    def reset(self) -> None:
        self.W = torch.zeros(self.d_val, self.d_key)

    def write(self, key: torch.Tensor, value: torch.Tensor, beta: float = 1.0) -> None:
        """Write one (key, value) with the delta rule. Key is L2-normalized."""
        k = _unit(key)
        read = self.W @ k                     # current stored value for k
        self.W = self.W + beta * torch.outer(value - read, k)

    def store(self, keys: torch.Tensor, values: torch.Tensor, beta: float = 1.0) -> None:
        """Store a set of pairs in ONE sequential pass (keys: (M,d_k), values (M,d_v))."""
        for k, v in zip(keys, values):
            self.write(k, v, beta)

    def read(self, query: torch.Tensor) -> torch.Tensor:
        """Recall the value associated with a (unit-normalized) query key."""
        return self.W @ _unit(query)

    def read_all(self, queries: torch.Tensor) -> torch.Tensor:
        """Recall for a batch of queries (M, d_k) -> (M, d_v)."""
        return _unit(queries) @ self.W.t()
