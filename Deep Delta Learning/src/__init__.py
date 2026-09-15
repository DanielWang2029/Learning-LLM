"""Minimal from-scratch reproduction of Deep Delta Learning (DDL).

Zhang, Liu, Wang & Gu, *"Deep Delta Learning"* (2026), arXiv:2601.00417.

Two faces of the same delta rule:

- ``DeltaAssociativeMemory``  the classic fast-weight associative memory that
  stores key->value bindings with ``W <- W + β (v - W k) kᵀ`` and recalls them.
  This is the underlying mechanism (Schlag et al. 2021; Yang et al. 2024) that
  the paper builds on.

- ``ddl_step`` / ``DDLBlock``  Deep Delta Learning's contribution: the *same*
  delta rule used as a **depth-wise residual interface**
  ``X_{l+1} = X_l + β_l k_l (v_l - k_lᵀ X_l)ᵀ`` (Eq. 2.2), which reads the
  residual state along a direction, compares it with a target, and writes back a
  gated rank-1 correction. Its signature property is the local error update
  ``e_post = (1 - β) e_pre`` (Eq. 2.4).
"""

from .delta_memory import DeltaAssociativeMemory
from .ddl import ddl_step, DDLBlock, readout

__all__ = ["DeltaAssociativeMemory", "ddl_step", "DDLBlock", "readout"]
