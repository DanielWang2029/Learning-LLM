"""Deep Delta Learning: the delta rule as a depth-wise residual interface.

Standard residual blocks update the residual stream additively:
``X_{l+1} = X_l + F_l(X_l)``. DDL (paper §2) instead parameterizes the update as
a *target-seeking rank-1 rewrite* along a learned unit direction:

    readout r = k_lᵀ X_l                                       # read state
    X_{l+1} = X_l + β_l k_l (v_l - k_lᵀ X_l)ᵀ                  (Eq. 2.2)

where ``k_l`` is a unit direction, ``v_l`` a target value, and ``β_l ∈ (0,2)`` a
shared gate. The signature local property (Eq. 2.4) is:

    e_pre  = k_lᵀ X_l   - v_l
    e_post = k_lᵀ X_{l+1} - v_l  =  (1 - β_l) · e_pre

so β=0 is the identity, β=1 exactly overwrites the selected readout, and
1<β<2 is an over-relaxed correction. Stacking these blocks lets each layer edit
one readout of the shared residual state — "deep" delta learning across depth.

State convention: ``X ∈ R^{d × d_v}`` (d = width, d_v = residual value channels;
d_v = 1 recovers an ordinary vector residual state).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _unit(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def readout(X: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """r = kᵀ X, the readout along direction k. X:(d,d_v), k:(d,) -> (d_v,)."""
    return X.t() @ _unit(k)


def ddl_step(X: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: float):
    """One DDL rewrite. Returns the new state X_{l+1}.

    X:(d,d_v), k:(d,), v:(d_v,), beta: scalar in (0,2).
    """
    k = _unit(k)
    r = X.t() @ k                     # (d_v,) current readout
    return X + beta * torch.outer(k, v - r)


class DDLBlock(nn.Module):
    """A learnable DDL residual block (Compress-Process-Rewrite, §2-3).

    Generates a unit read/write direction ``k``, a target value ``v``, and a
    gate ``β ∈ (0,2)`` from the compressed residual state, then applies the
    rank-1 delta rewrite. ``d_v`` value channels give an expanded residual state
    without widening the compute path.
    """

    def __init__(self, d_model: int, d_val: int = 1) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_val = d_val
        self.norm = nn.LayerNorm(d_model)
        self.to_k = nn.Linear(d_model, d_model)
        self.to_v = nn.Linear(d_model, d_val)
        self.to_beta = nn.Linear(d_model, 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X: (batch, d_model, d_val). Compress -> generate k,v,β -> rewrite."""
        ctx = self.norm(X.mean(dim=-1))          # compress value channels: (B, d_model)
        k = _unit(self.to_k(ctx))                # (B, d_model)
        v = self.to_v(ctx)                       # (B, d_val)
        beta = 2.0 * torch.sigmoid(self.to_beta(ctx))  # (B,1) in (0,2)
        r = torch.einsum("bd,bde->be", k, X)     # readout (B, d_val)
        delta = beta * (v - r)                   # (B, d_val)
        return X + torch.einsum("bd,be->bde", k, delta)
