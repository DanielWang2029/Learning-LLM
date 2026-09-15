"""From-scratch reproduction of Nemotron 3 Super's hybrid architecture.

Nemotron 3 Super (NVIDIA, 2026) is a hybrid **Mamba-Attention** MoE: most layers
are Mamba-2-style state-space (SSM) layers — linear-time, constant-memory
sequence mixers — interleaved with a few full self-attention layers for global
interaction (plus LatentMoE). This folder implements a tiny selective SSM layer,
a causal attention layer, and a hybrid stack that alternates them.

Public API:

- ``SelectiveSSM``  a Mamba-2-style selective state-space layer (linear time)
- ``CausalAttention`` standard causal multi-head attention (quadratic)
- ``HybridModel``   a stack that interleaves SSM and attention layers
"""

from .ssm import SelectiveSSM
from .attention import CausalAttention
from .model import HybridModel

__all__ = ["SelectiveSSM", "CausalAttention", "HybridModel"]
