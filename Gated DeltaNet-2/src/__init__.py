"""Minimal from-scratch reproduction of Gated DeltaNet-2.

Hatamizadeh, Choi & Kautz, *"Gated DeltaNet-2: Decoupling Erase and Write in
Linear Attention"* (2026), arXiv:2605.22791.

Public API mirrors the paper's core operator (Section 3):

- ``GatedDeltaRule2``   the recurrent linear-attention state update (Eq. 10)
- ``GatedDeltaNet2``    a tiny sequence model that uses the operator as its
                        token mixer for associative recall.
- ``additive_linear_attention``  the plain linear-attention recurrence (Eq. 1)
                        used as a "no-erase" baseline in the demo.
"""

from .gated_deltanet2 import (
    GatedDeltaRule2,
    GatedDeltaNet2,
    additive_linear_attention,
)

__all__ = [
    "GatedDeltaRule2",
    "GatedDeltaNet2",
    "additive_linear_attention",
]
