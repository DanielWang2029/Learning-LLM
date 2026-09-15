"""DeepSeek Sparse Attention (DSA), the core of GLM-5 (paper Section 2.1.1).

The philosophy of DSA is to replace dense O(L^2) attention with a dynamic,
content-based *fine-grained selection* mechanism. A lightweight "lightning
indexer" cheaply predicts which keys matter for each query; only the top-k
keys are then attended to with ordinary softmax attention. Because k << L,
the expensive attention is computed over a tiny fraction of the keys while
long-range dependencies are preserved (the indexer "looks" at content, unlike
a fixed sliding window).

Everything here is intentionally tiny and CPU-only.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def full_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dense scaled dot-product attention -- the O(L^2) reference.

    Shapes: q,k,v = (batch, seq, d). Returns (output, attn_weights) where
    ``attn_weights`` is (batch, seq_q, seq_k). Every query scores *every* key.
    """
    d = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    return out, attn


class LightningIndexer(nn.Module):
    """Cheap per-(query, key) importance scorer (DSA "lightning indexer").

    The indexer runs in a small ``d_index`` dimension with a few light heads,
    so scoring all L keys costs far less than a full attention head. It scores
    a query-key pair with a ReLU inner product summed over heads:

        I[t, s] = sum_h  w_h * ReLU( qI[t,h] . kI[s,h] )

    Top-k over ``I[t, :]`` then selects the keys that get real attention. The
    indexer is trained (see the demo) to make its top-k agree with where the
    true dense attention actually puts its mass.
    """

    def __init__(self, d_model: int, d_index: int = 16, n_heads: int = 2) -> None:
        super().__init__()
        self.d_index = d_index
        self.n_heads = n_heads
        self.wq = nn.Linear(d_model, d_index * n_heads, bias=False)
        self.wk = nn.Linear(d_model, d_index * n_heads, bias=False)
        # Per-head positive weights on the ReLU similarity contributions.
        self.head_weight = nn.Parameter(torch.zeros(n_heads))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq, d_model) -> index scores (batch, seq_q, seq_k)."""
        b, l, _ = x.shape
        qi = self.wq(x).view(b, l, self.n_heads, self.d_index)
        ki = self.wk(x).view(b, l, self.n_heads, self.d_index)
        # sim[b, h, t, s] = qi[b,t,h,:] . ki[b,s,h,:]
        sim = torch.einsum("bthd,bshd->bhts", qi, ki)
        sim = F.relu(sim)
        w = F.softplus(self.head_weight)  # keep head weights positive
        scores = torch.einsum("h,bhts->bts", w, sim)
        return scores


def dsa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    index_scores: torch.Tensor,
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top-k sparse attention driven by the indexer's ``index_scores``.

    For each query we keep only the ``top_k`` highest-scored keys and run
    ordinary softmax attention over just those. Returns (output, selected_idx)
    with selected_idx of shape (batch, seq_q, top_k).
    """
    b, l, d = q.shape
    top_k = min(top_k, l)
    sel = index_scores.topk(top_k, dim=-1).indices  # (b, lq, top_k)

    # Gather the selected keys/values for every query.
    k_exp = k.unsqueeze(1).expand(b, l, l, d)  # (b, lq, lk, d)
    v_exp = v.unsqueeze(1).expand(b, l, l, d)
    idx = sel.unsqueeze(-1).expand(b, l, top_k, d)
    k_sel = torch.gather(k_exp, 2, idx)  # (b, lq, top_k, d)
    v_sel = torch.gather(v_exp, 2, idx)

    scores = torch.einsum("bqd,bqkd->bqk", q, k_sel) / math.sqrt(d)
    attn = F.softmax(scores, dim=-1)
    out = torch.einsum("bqk,bqkd->bqd", attn, v_sel)
    return out, sel


def attention_mass_recall(
    full_attn: torch.Tensor, selected_idx: torch.Tensor
) -> torch.Tensor:
    """Fraction of the *true* dense attention mass captured by the top-k keys.

    This is the key DSA quality metric: if the indexer selects the keys where
    dense attention already concentrated, recall approaches 1.0 and the sparse
    output is nearly lossless. Random selection recovers only ~ top_k / L.
    """
    picked = torch.gather(full_attn, 2, selected_idx)  # (b, lq, top_k)
    return picked.sum(dim=-1).mean()
