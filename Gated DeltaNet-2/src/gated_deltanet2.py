"""Gated Delta Rule-2: linear attention with *decoupled* erase and write gates.

This file implements the central operator of Gated DeltaNet-2 (paper Section 3).

Background (Section 2). Linear attention keeps a fixed-size matrix state
``S ∈ R^{d_k × d_v}`` and reads it with the query:

    additive:      S_t = S_{t-1} + k_t v_tᵀ                 (Eq. 1, no forgetting)
    DeltaNet:      S_t = (I - β_t k_t k_tᵀ) S_{t-1} + β_t k_t v_tᵀ   (Eq. 5)
    Gated DeltaNet:S_t = α_t (I - β_t k_t k_tᵀ) S_{t-1} + β_t k_t v_tᵀ (Eq. 6)

In all of the above the *active edit* uses a single scalar β_t to control two
different things at once: how much old content to ERASE (a key-side decision)
and how much new content to WRITE (a value-side decision).

Gated Delta Rule-2 (Eq. 8-10) breaks that tie with two channel-wise gates:

    e_t = b_t ⊙ k_t         b_t ∈ [0,1]^{d_k}  is the channel-wise ERASE gate
    z_t = w_t ⊙ v_t         w_t ∈ [0,1]^{d_v}  is the channel-wise WRITE gate
    S̄_t = D_t S_{t-1}       D_t = diag(α_t), channel-wise decay (from KDA)
    r_t = S̄_tᵀ e_t          read the old content along the gated erase direction
    S_t = S̄_t + k_t (z_t - r_t)ᵀ                                   (Eq. 9)
        = (I - k_t (b_t⊙k_t)ᵀ) D_t S_{t-1} + k_t (w_t⊙v_t)ᵀ        (Eq. 10)
    o_t = S_tᵀ q_t

Setting b_t = w_t = β_t·1 recovers KDA (scalar active gate); further tying the
decay recovers Gated DeltaNet.  The demo shows why the *decoupling* matters for
overwriting one association without scrambling the rest.

Everything here is the plain O(L) recurrence (a Python loop over time). The
paper's chunkwise/WY kernel (Section 3.3) is a training-speed optimization that
computes exactly the same function; we keep the recurrence for clarity and
because sequences in the demo are short.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


class GatedDeltaRule2(nn.Module):
    """The Gated Delta Rule-2 recurrent operator (paper Eq. 8-10).

    Args:
        d_model:  input/output feature dimension.
        d_key:    key/query dimension d_k of the associative state.
        d_val:    value dimension d_v of the associative state.
        decouple: if True, use independent channel-wise erase gate ``b`` and
                  write gate ``w`` (the paper's contribution). If False, tie
                  both to a single scalar gate ``β`` per token (the KDA-style
                  baseline this paper argues against).
    """

    def __init__(
        self,
        d_model: int,
        d_key: int = 32,
        d_val: int = 32,
        decouple: bool = True,
    ) -> None:
        super().__init__()
        self.d_key = d_key
        self.d_val = d_val
        self.decouple = decouple

        # q, k, v projections (Section 3.5 uses linear + short conv + SiLU; we
        # keep just the linear projection for a minimal CPU model).
        self.w_q = nn.Linear(d_model, d_key)
        self.w_k = nn.Linear(d_model, d_key)
        self.w_v = nn.Linear(d_model, d_val)

        # Channel-wise decay α_t = exp(g_t) with g_t <= 0 (Eq. 12).
        self.w_decay = nn.Linear(d_model, d_key)
        self.a_decay = nn.Parameter(torch.zeros(d_key))

        if decouple:
            # Separate erase (key axis) and write (value axis) gates (Eq. 11).
            self.w_erase = nn.Linear(d_model, d_key)
            self.w_write = nn.Linear(d_model, d_val)
        else:
            # Tied scalar active gate β_t shared by erase and write (KDA-style).
            self.w_beta = nn.Linear(d_model, 1)

        # Output normalization + gate + projection (Section 3.5).
        self.out_norm = nn.LayerNorm(d_val)
        self.w_out_gate = nn.Linear(d_model, d_val)
        self.w_out = nn.Linear(d_val, d_model)

    def gates(self, x: torch.Tensor):
        """Return (alpha, b, w) gate tensors for an input batch x: (B, T, d_model)."""
        alpha = torch.exp(
            -torch.exp(self.a_decay) * F.softplus(self.w_decay(x))
        )  # (B, T, d_key) in (0, 1]
        if self.decouple:
            b = torch.sigmoid(self.w_erase(x))  # (B, T, d_key)
            w = torch.sigmoid(self.w_write(x))  # (B, T, d_val)
        else:
            beta = torch.sigmoid(self.w_beta(x))  # (B, T, 1)
            b = beta.expand(-1, -1, self.d_key)
            w = beta.expand(-1, -1, self.d_val)
        return alpha, b, w

    def forward(self, x: torch.Tensor, return_trace: bool = False):
        """Run the recurrence over a sequence.

        x: (B, T, d_model). Returns y: (B, T, d_model). If ``return_trace`` is
        set, also returns a dict with per-step read magnitudes for inspection.
        """
        B, T, _ = x.shape
        q = _l2norm(self.w_q(x))          # (B, T, d_key)
        k = _l2norm(self.w_k(x))          # (B, T, d_key)
        v = self.w_v(x)                   # (B, T, d_val)
        alpha, b, w = self.gates(x)

        S = x.new_zeros(B, self.d_key, self.d_val)  # fixed-size state
        outputs = []
        erase_norms, write_norms = [], []

        for t in range(T):
            k_t = k[:, t]                 # (B, d_key)
            q_t = q[:, t]
            v_t = v[:, t]
            a_t = alpha[:, t]             # (B, d_key)
            e_t = b[:, t] * k_t           # gated erase direction (B, d_key)
            z_t = w[:, t] * v_t           # gated write target   (B, d_val)

            # Apply channel-wise decay to the key axis (rows) of the state.
            S = a_t.unsqueeze(-1) * S     # S̄ = D_t S_{t-1}

            # Read the value currently stored along the gated erase direction.
            r_t = torch.einsum("bi,bij->bj", e_t, S)     # S̄ᵀ e_t  (B, d_val)

            # Write a rank-1 correction toward (z_t - r_t) along k_t (Eq. 9).
            delta = z_t - r_t
            S = S + torch.einsum("bi,bj->bij", k_t, delta)

            # Read out with the query: o_t = S_tᵀ q_t.
            o_t = torch.einsum("bi,bij->bj", q_t, S)      # (B, d_val)
            outputs.append(o_t)

            if return_trace:
                erase_norms.append(r_t.norm(dim=-1))
                write_norms.append(z_t.norm(dim=-1))

        o = torch.stack(outputs, dim=1)                   # (B, T, d_val)
        o = self.out_norm(o) * F.silu(self.w_out_gate(x))
        y = self.w_out(o)

        if return_trace:
            trace = {
                "erase_read_norm": torch.stack(erase_norms, dim=1),
                "write_norm": torch.stack(write_norms, dim=1),
            }
            return y, trace
        return y


class GatedDeltaNet2(nn.Module):
    """A tiny sequence model: embedding -> GatedDeltaRule2 mixer -> readout.

    Used in the demo for a key-value associative-recall-with-overwrite task.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        d_key: int = 32,
        d_val: int = 32,
        decouple: bool = True,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.mixer = GatedDeltaRule2(d_model, d_key, d_val, decouple=decouple)
        self.norm = nn.LayerNorm(d_model)
        self.readout = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        x = x + self.mixer(self.norm(x))   # residual around the mixer
        return self.readout(x)


@torch.no_grad()
def additive_linear_attention(
    k: torch.Tensor, v: torch.Tensor, q: torch.Tensor
) -> torch.Tensor:
    """Plain linear attention recurrence S_t = S_{t-1} + k_t v_tᵀ (Eq. 1).

    No erase/decay: every write is *added* to the state and nothing is removed.
    This is the "no-erase" baseline used in the demo to show interference when a
    key is overwritten. Shapes: k,q (B, T, d_k); v (B, T, d_v).
    Returns reads o_t = S_tᵀ q_t of shape (B, T, d_v).
    """
    B, T, d_k = k.shape
    d_v = v.shape[-1]
    S = k.new_zeros(B, d_k, d_v)
    outs = []
    for t in range(T):
        S = S + torch.einsum("bi,bj->bij", k[:, t], v[:, t])
        outs.append(torch.einsum("bi,bij->bj", q[:, t], S))
    return torch.stack(outs, dim=1)
