"""Kimi Delta Attention (KDA), Kimi K3 Section 2.1.1.

KDA extends the delta-rule recurrence with a *channel-wise forget gate*. For a
single head with query/key qₜ,kₜ ∈ R^{dk}, value vₜ ∈ R^{dv}, and recurrent
state Sₜ ∈ R^{dk×dv} (Eq. 1 of the paper):

    Sₜ = (I − βₜ kₜ kₜᵀ) · Diag(αₜ) · Sₜ₋₁ + βₜ kₜ vₜᵀ
    õₜ = Sₜᵀ qₜ

where αₜ ∈ (0,1)^{dk} is the per-channel one-step retention (forget gate) and
βₜ ∈ (0,1) is the delta-rule write strength. Rearranging shows the delta rule:

    S̃ = Diag(αₜ)·Sₜ₋₁                       # decay the memory
    read = kₜᵀ S̃                            # what memory currently predicts for kₜ
    Sₜ = S̃ + βₜ · kₜ (vₜ − read)ᵀ           # write the *correction* only

The state is fixed size, so the whole layer is O(L) in time and O(1) in memory
per step — the linear-attention property. Following the paper, q/k/v come from a
ShortConv + Swish (and q,k are L2-normalized) so each position can also see its
immediate neighbor, which is what makes associative key→value binding possible.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _short_conv(x: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
    """Depthwise *causal* 1-D conv over the sequence dim. x: (B, L, C)."""
    b, l, c = x.shape
    xt = x.transpose(1, 2)  # (B, C, L)
    k = conv.kernel_size[0]
    xt = F.pad(xt, (k - 1, 0))  # left-pad -> causal
    xt = conv(xt)
    return xt.transpose(1, 2)  # (B, L, C)


class KDA(nn.Module):
    """Gated delta-rule linear attention (single- or multi-head)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 2,
        head_dim: int = 32,
        alpha_min: float = 0.90,
        conv_kernel: int = 3,
    ) -> None:
        super().__init__()
        self.h = n_heads
        self.dk = head_dim
        self.dv = head_dim
        self.alpha_min = alpha_min
        inner = n_heads * head_dim

        self.w_q = nn.Linear(d_model, inner, bias=False)
        self.w_k = nn.Linear(d_model, inner, bias=False)
        self.w_v = nn.Linear(d_model, inner, bias=False)
        self.w_beta = nn.Linear(d_model, n_heads)          # write strength β
        self.w_alpha = nn.Linear(d_model, inner)           # per-channel decay α
        self.w_o = nn.Linear(inner, d_model, bias=False)

        self.conv_q = nn.Conv1d(inner, inner, conv_kernel, groups=inner, bias=False)
        self.conv_k = nn.Conv1d(inner, inner, conv_kernel, groups=inner, bias=False)
        self.conv_v = nn.Conv1d(inner, inner, conv_kernel, groups=inner, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        H, dk, dv = self.h, self.dk, self.dv

        q = F.silu(_short_conv(self.w_q(x), self.conv_q))
        k = F.silu(_short_conv(self.w_k(x), self.conv_k))
        v = F.silu(_short_conv(self.w_v(x), self.conv_v))

        q = q.view(b, l, H, dk)
        k = k.view(b, l, H, dk)
        v = v.view(b, l, H, dv)
        # L2-normalize queries and keys (paper: L2Norm on q, k).
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        beta = torch.sigmoid(self.w_beta(x)).view(b, l, H, 1)          # (B,L,H,1)
        # Lower-bounded per-channel retention: α ∈ (alpha_min, 1).
        alpha = self.alpha_min + (1 - self.alpha_min) * torch.sigmoid(
            self.w_alpha(x)
        ).view(b, l, H, dk)

        # Linear-time recurrence over the sequence (fixed-size state S).
        S = x.new_zeros(b, H, dk, dv)
        outs = []
        for t in range(l):
            a_t = alpha[:, t]                       # (B,H,dk)
            k_t = k[:, t]                           # (B,H,dk)
            v_t = v[:, t]                           # (B,H,dv)
            q_t = q[:, t]                           # (B,H,dk)
            b_t = beta[:, t]                        # (B,H,1)

            S = a_t.unsqueeze(-1) * S               # Diag(α)·S : decay memory
            read = torch.einsum("bhd,bhde->bhe", k_t, S)   # kᵀ S̃
            delta = v_t - read                              # delta-rule error
            S = S + b_t.unsqueeze(-1) * torch.einsum(
                "bhd,bhe->bhde", k_t, delta
            )                                               # write correction
            o_t = torch.einsum("bhde,bhd->bhe", S, q_t)     # Sᵀ q
            outs.append(o_t)

        o = torch.stack(outs, dim=1).reshape(b, l, H * dv)
        return self.w_o(o)


class KDAModel(nn.Module):
    """Tiny token model: embedding → KDA block(s) → LM head (demo only)."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 2,
        head_dim: int = 32,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [KDA(d_model, n_heads, head_dim) for _ in range(n_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.embed(tokens)
        for layer, norm in zip(self.layers, self.norms):
            h = h + layer(norm(h))  # pre-norm residual
        return self.head(h)
