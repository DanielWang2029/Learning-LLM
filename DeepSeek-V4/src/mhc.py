"""Manifold-Constrained Hyper-Connections (mHC), DeepSeek-V4 Section 2.2.

Hyper-Connections widen the residual stream to n_hc parallel copies and mix
them each layer with a matrix B ∈ R^{n_hc×n_hc}. mHC's core idea is to constrain
B to the manifold of **doubly-stochastic** matrices (the Birkhoff polytope):
every row and column sums to 1 and entries are non-negative. Such a B is
non-expansive (its spectral norm is ≤ 1), which keeps deep residual stacks
numerically stable.

The projection onto that manifold is done with Sinkhorn normalization — the
practical realization the paper uses (iterating row/column normalization of a
non-negative matrix converges to a doubly-stochastic one).
"""

from __future__ import annotations

import torch


def doubly_stochastic(raw: torch.Tensor, iters: int = 20, eps: float = 1e-8) -> torch.Tensor:
    """Project a raw square matrix onto the doubly-stochastic manifold.

    ``raw`` is first mapped to non-negative entries via softplus, then Sinkhorn
    row/column normalization is applied for ``iters`` steps (paper uses t_max=20).
    Returns a matrix whose rows and columns each sum to ~1.
    """
    assert raw.dim() == 2 and raw.size(0) == raw.size(1), "expected square matrix"
    m = torch.nn.functional.softplus(raw) + eps
    for _ in range(iters):
        m = m / m.sum(dim=1, keepdim=True)  # rows sum to 1
        m = m / m.sum(dim=0, keepdim=True)  # cols sum to 1
    return m
