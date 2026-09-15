"""From-scratch reproduction of DeepSeek-V4's distinctive training ingredients.

DeepSeek-V4 (DeepSeek-AI, 2026) trains an efficient million-token model with,
among other things:

- the **Muon optimizer** (Section 2.4): orthogonalize the momentum update via
  *hybrid Newton–Schulz* iterations before applying it, and
- **Manifold-Constrained Hyper-Connections (mHC)** (Section 2.2): constrain the
  residual-mixing matrix to the manifold of doubly-stochastic matrices so the
  residual mapping is non-expansive and training stays stable.

Public API:

- ``Muon``                    the optimizer (Muon for matrices, Adam for the rest)
- ``newton_schulz``           hybrid Newton–Schulz orthogonalization
- ``doubly_stochastic``       Sinkhorn projection used by mHC
"""

from .muon import Muon, newton_schulz
from .mhc import doubly_stochastic

__all__ = ["Muon", "newton_schulz", "doubly_stochastic"]
