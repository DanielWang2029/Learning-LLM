"""A minimal Mamba-2-style selective state-space (SSM) layer.

State-space models mix a sequence with a *linear recurrence* over a fixed-size
hidden state, giving O(L) time and O(1) memory per step (no L×L matrix). The
"selective" part (Mamba) makes the recurrence *input-dependent*: the decay dt,
the input gate B and the output gate C are all functions of the current token,
so the model can choose what to remember and what to forget.

Recurrence (per channel, hidden state h ∈ R^{d_state}):

    aₜ = exp(−Δₜ · exp(A))              # input-dependent decay in (0,1)
    Hₜ = aₜ · Hₜ₋₁ + xₜ · Bₜ            # write the current token into the state
    yₜ = ⟨Hₜ, Cₜ⟩ + D · xₜ             # read the state out

This is the recurrent form of Mamba-2 / SSD, kept intentionally tiny.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    def __init__(self, d_model: int, d_state: int = 8, expand: int = 1) -> None:
        super().__init__()
        self.d_inner = d_model * expand
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)   # -> x and gate z
        self.bc_proj = nn.Linear(self.d_inner, 2 * d_state)   # -> Bₜ, Cₜ (selective)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)  # -> Δₜ (selective)
        # A is a per-channel base decay rate (learned, parameterized in log space).
        self.A_log = nn.Parameter(torch.zeros(self.d_inner))
        self.D = nn.Parameter(torch.ones(self.d_inner))       # skip connection
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)          # (B,L,d_inner) each
        x_in = F.silu(x_in)

        bc = self.bc_proj(x_in)
        B, C = bc.chunk(2, dim=-1)             # (B,L,d_state) each
        dt = F.softplus(self.dt_proj(x_in))    # (B,L,d_inner), positive
        A = torch.exp(self.A_log)              # (d_inner,), positive

        # Linear recurrence over the sequence with a fixed-size hidden state.
        H = x.new_zeros(b, self.d_inner, self.d_state)
        ys = []
        for t in range(l):
            a_t = torch.exp(-dt[:, t] * A)                    # (B,d_inner) decay
            H = a_t.unsqueeze(-1) * H + x_in[:, t].unsqueeze(-1) * B[:, t].unsqueeze(1)
            y_t = (H * C[:, t].unsqueeze(1)).sum(-1) + self.D * x_in[:, t]
            ys.append(y_t)
        y = torch.stack(ys, dim=1)             # (B,L,d_inner)

        y = y * F.silu(z)                      # Mamba output gate
        return self.out_proj(y)
