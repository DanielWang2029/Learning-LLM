"""A tiny transformer whose feed-forward sub-layer is a Switch MoE layer.

The Switch paper swaps the dense FFN of a transformer block for a sparse top-1
MoE (:class:`~src.moe.SwitchFFN`). We keep everything else minimal: token
embeddings, the Switch FFN sub-layer (pre-norm + residual, exactly where a
normal FFN would go), and a classification head. Routing is per token.

To make *emergent specialization* visible in a tiny model we split the token
into its two factors (see data/generate_data.py):

    type  g  = token %  n_types      (which kind of token)
    value v  = token // n_types      (its payload)

The experts only ever see the **value** embedding, while the router sees the
**type**. Because the correct label depends on both, an expert that is handed
two different types sees the *same* value mapped to *conflicting* labels and
cannot fit them. The only low-loss solution is for the router to send each type
to its own expert — i.e. the experts must specialize. This mirrors why real
MoE routers specialize (interference between sub-tasks + limited per-expert
capacity), just made sharp enough to see in seconds on CPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .moe import SwitchFFN


class TinySwitchModel(nn.Module):
    def __init__(self, n_types: int, values_per_type: int, n_classes: int,
                 d_model: int = 32, d_ff: int = 64, n_experts: int = 4,
                 alpha: float = 0.01) -> None:
        super().__init__()
        self.n_types = n_types
        self.value_emb = nn.Embedding(values_per_type, d_model)  # experts see this
        self.type_emb = nn.Embedding(n_types, d_model)           # router sees this
        self.ln = nn.LayerNorm(d_model)
        self.switch = SwitchFFN(d_model, d_ff, n_experts, alpha=alpha)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None):
        g = tokens % self.n_types           # type
        v = tokens // self.n_types          # value
        xv = self.value_emb(v)              # expert input (no type information)
        route_in = self.type_emb(g)         # router input (type only)

        ff, route = self.switch(self.ln(xv), route_input=route_in)
        x = xv + ff                          # residual on the value stream
        logits = self.head(x)

        loss = None
        if targets is not None:
            ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                 targets.reshape(-1))
            loss = ce + route.aux_loss
        return logits, route, loss
