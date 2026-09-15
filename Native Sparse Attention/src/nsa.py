"""Native Sparse Attention (NSA), paper §3.

NSA replaces full attention with three parallel branches whose outputs are
merged by a learned per-query **gate** (paper Eq. 5 / Fig. 2):

1. **Compression** (§3.3.1) — keys/values are pooled blockwise into a small set
   of coarse tokens, giving cheap *global* context.
2. **Selection** (§3.3.2) — the compression attention scores are reused to pick
   the top-``n`` most important blocks; their *fine-grained* keys/values are
   attended to. This is the natively-trainable sparse core.
3. **Sliding window** (§3.3.3) — a small local window of the most recent tokens,
   so the model never has to "re-learn" local patterns through the sparse paths.

Output(q) = g_cmp · O_cmp + g_slc · O_slc + g_win · O_win

Everything is causal. This implementation favours clarity over hardware
efficiency: the branches are computed with masked dense attention (fine at the
small sizes used in the demo), and the number of key positions each branch
*would* touch is reported separately to show the sparsity.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Softmax over the last dim where ``mask`` is True; fully-masked rows -> 0."""
    neg = torch.finfo(scores.dtype).min
    scores = scores.masked_fill(~mask, neg)
    scores = scores - scores.max(dim=-1, keepdim=True).values
    w = torch.exp(scores)
    w = w * mask
    denom = w.sum(dim=-1, keepdim=True)
    return torch.where(denom > 0, w / denom.clamp_min(1e-9), torch.zeros_like(w))


class NSAAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int = 2,
        block: int = 4,       # compression / selection block size
        n_select: int = 2,    # number of fine blocks selected per query
        window: int = 8,      # sliding-window size
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.h = num_heads
        self.dk = d_model // num_heads
        self.block = block
        self.n_select = n_select
        self.window = window

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        # Learned pooling for compression: mean-pool then a small linear.
        self.cmp_k = nn.Linear(self.dk, self.dk)
        self.cmp_v = nn.Linear(self.dk, self.dk)
        # Gate: query -> 3 branch weights.
        self.gate = nn.Linear(d_model, 3)

        # Diagnostics from the last forward pass (for logging / visualization).
        self.last_stats: Dict[str, float] = {}

    def _split(self, x):
        b, t, _ = x.shape
        return x.view(b, t, self.h, self.dk).transpose(1, 2)  # (b,h,t,dk)

    def forward(self, x: torch.Tensor, collect: bool = False) -> torch.Tensor:
        b, t, d = x.shape
        H, dk, l = self.h, self.dk, self.block
        device = x.device

        q = self._split(self.w_q(x))
        k = self._split(self.w_k(x))
        v = self._split(self.w_v(x))
        scale = 1.0 / math.sqrt(dk)

        pos = torch.arange(t, device=device)
        nb = (t + l - 1) // l  # number of blocks

        # ---- pad to a whole number of blocks for pooling
        pad = nb * l - t
        kk = F.pad(k, (0, 0, 0, pad))  # (b,h,t+pad,dk)
        vv = F.pad(v, (0, 0, 0, pad))
        kb = kk.view(b, H, nb, l, dk).mean(dim=3)   # blockwise mean pool
        vb = vv.view(b, H, nb, l, dk).mean(dim=3)
        kcmp = self.cmp_k(kb)                        # (b,h,nb,dk)
        vcmp = self.cmp_v(vb)

        block_start = torch.arange(nb, device=device) * l
        block_end = block_start + l - 1              # inclusive last index in block
        # A block is usable by query t (causally, at block granularity) only if
        # the whole block lies within [0, t].
        cmp_allow = (block_end.unsqueeze(0) <= pos.unsqueeze(1))  # (t, nb)

        # ---- Branch 1: compression attention (coarse, global)
        s_cmp = torch.einsum("bhtd,bhjd->bhtj", q, kcmp) * scale  # (b,h,t,nb)
        cmp_mask = cmp_allow.view(1, 1, t, nb).expand(b, H, t, nb)
        a_cmp = _safe_softmax(s_cmp, cmp_mask)
        o_cmp = torch.einsum("bhtj,bhjd->bhtd", a_cmp, vcmp)

        # ---- Branch 2: selection (reuse compression scores to pick top-n blocks)
        importance = a_cmp  # (b,h,t,nb); higher = more relevant block
        n_sel = min(self.n_select, nb)
        top_idx = importance.topk(n_sel, dim=-1).indices           # (b,h,t,n_sel)
        block_sel = torch.zeros(b, H, t, nb, dtype=torch.bool, device=device)
        block_sel.scatter_(-1, top_idx, True)
        block_sel = block_sel & cmp_mask                           # only usable blocks
        # expand selected blocks to fine key positions
        pos_block = (pos // l)                                     # (t_keys,) block id of each key
        sel_fine = block_sel[..., pos_block[:t]]                   # (b,h,t,t) via gather on block dim
        causal = (pos.unsqueeze(1) >= pos.unsqueeze(0)).view(1, 1, t, t)  # key<=query
        sel_mask = sel_fine & causal
        s_fine = torch.einsum("bhtd,bhsd->bhts", q, k) * scale
        a_slc = _safe_softmax(s_fine, sel_mask)
        o_slc = torch.einsum("bhts,bhsd->bhtd", a_slc, v)

        # ---- Branch 3: sliding window (local)
        win_mask = causal & ((pos.unsqueeze(1) - pos.unsqueeze(0)) < self.window).view(1, 1, t, t)
        a_win = _safe_softmax(s_fine, win_mask)
        o_win = torch.einsum("bhts,bhsd->bhtd", a_win, v)

        # ---- Gate & combine
        g = torch.softmax(self.gate(x), dim=-1)          # (b,t,3)
        g = g.unsqueeze(1)                               # (b,1,t,3) broadcast over heads
        out = g[..., 0:1] * o_cmp + g[..., 1:2] * o_slc + g[..., 2:3] * o_win
        out = out.transpose(1, 2).contiguous().view(b, t, d)

        if collect:
            with torch.no_grad():
                # average gate weights over batch & positions
                gm = g.mean(dim=(0, 1, 2)).tolist()
                # Key positions each query "touches" = distinct fine positions
                # (selected blocks ∪ sliding window) + compressed coarse tokens.
                fine_attended = (sel_mask | win_mask).any(dim=1).float().sum(dim=-1)  # (b,t)
                cmp_attended = cmp_mask.any(dim=1).float().sum(dim=-1)                # (b,t)
                nsa_positions = fine_attended + cmp_attended                          # (b,t)
                full_positions = (pos + 1).float().unsqueeze(0).expand(b, t)          # (b,t)
                # A full-context query (last position) sees the whole sequence under
                # full attention; NSA still only touches its sparse budget.
                self.last_stats = {
                    "gate_cmp": gm[0], "gate_slc": gm[1], "gate_win": gm[2],
                    "nsa_positions_mean": float(nsa_positions.mean().item()),
                    "full_positions_mean": float(full_positions.mean().item()),
                    "nsa_positions_last": float(nsa_positions[:, -1].mean().item()),
                    "full_positions_last": float(full_positions[:, -1].mean().item()),
                    "num_blocks": float(nb),
                }
        return self.w_o(out)


class FullAttention(nn.Module):
    """Standard causal multi-head attention baseline."""

    def __init__(self, d_model: int, num_heads: int = 2) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.h = num_heads
        self.dk = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, collect: bool = False) -> torch.Tensor:
        b, t, d = x.shape
        H, dk = self.h, self.dk
        q = x.new_empty(0)  # placeholder to appease linters
        q = self.w_q(x).view(b, t, H, dk).transpose(1, 2)
        k = self.w_k(x).view(b, t, H, dk).transpose(1, 2)
        v = self.w_v(x).view(b, t, H, dk).transpose(1, 2)
        scale = 1.0 / math.sqrt(dk)
        scores = torch.einsum("bhtd,bhsd->bhts", q, k) * scale
        pos = torch.arange(t, device=x.device)
        causal = (pos.unsqueeze(1) >= pos.unsqueeze(0)).view(1, 1, t, t)
        scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
        a = torch.softmax(scores, dim=-1)
        out = torch.einsum("bhts,bhsd->bhtd", a, v).transpose(1, 2).contiguous().view(b, t, d)
        return self.w_o(out)
