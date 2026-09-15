"""From-scratch reproduction of Kimi K3's core sequence mixer.

Kimi K3 (Kimi Team, 2026) uses **Kimi Delta Attention (KDA)** — a *gated
delta-rule linear attention* — for efficient long-context mixing (Section
2.1.1). KDA keeps a fixed-size recurrent state and updates it with the delta
rule plus a channel-wise forget gate, giving linear-time, constant-memory
attention (no L×L matrix).

Public API:

- ``KDA``      the gated delta-rule linear-attention layer
- ``KDAModel`` a tiny token model built from KDA layers (for the demo)
"""

from .kda import KDA, KDAModel

__all__ = ["KDA", "KDAModel"]
