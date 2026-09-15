"""FlashAttention: tiled attention with an online (streaming) softmax.

Dao et al. (2022) observe that standard attention is bottlenecked by *memory
I/O*: it materializes the full N x N score matrix S = QKᵀ in slow HBM. Flash-
Attention never forms that matrix. It walks over blocks of keys/values and keeps
a running softmax using two scalars per query row — the running max ``m`` and
the running normalizer ``l`` — rescaling the partial output on the fly
(paper §3.1, Algorithm 1).

This module implements both, from scratch, so the demo can show (a) the tiled
output is numerically identical to the naive one, and (b) the tiled version
never allocates an N x N matrix.
"""

from __future__ import annotations

import math

import torch


def naive_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                    causal: bool = False):
    """Textbook attention. Materializes the full N x N score matrix.

    Q, K, V: (N, d). Returns (output (N, d), peak_score_elems).
    ``peak_score_elems`` is the number of floats held at once in the score
    matrix — the quantity FlashAttention is designed to avoid.
    """
    N, d = Q.shape
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d)  # (N, N)  <-- the problem
    if causal:
        mask = torch.tril(torch.ones(N, N, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = attn @ V
    return out, scores.numel()


def flash_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                    block_q: int = 32, block_k: int = 32, causal: bool = False):
    """Tiled attention with the online softmax (paper Algorithm 1).

    Processes the attention in ``block_q`` x ``block_k`` tiles, keeping only a
    running max and running normalizer per query row, so the largest score
    tensor ever in memory is a single ``block_q`` x ``block_k`` tile.

    Q, K, V: (N, d). Returns (output (N, d), peak_score_elems).
    """
    N, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    O = torch.zeros(N, d)
    peak_tile = 0

    for qs in range(0, N, block_q):
        qe = min(qs + block_q, N)
        Qi = Q[qs:qe]  # (bq, d)
        bq = qe - qs

        # Running statistics for this query block (online softmax state).
        m_i = torch.full((bq,), float("-inf"))
        l_i = torch.zeros(bq)
        O_i = torch.zeros(bq, d)

        for ks in range(0, N, block_k):
            ke = min(ks + block_k, N)
            Kj, Vj = K[ks:ke], V[ks:ke]  # (bk, d)

            # The ONLY score tensor materialized: a single small tile.
            S_ij = (Qi @ Kj.transpose(-2, -1)) * scale  # (bq, bk)
            peak_tile = max(peak_tile, S_ij.numel())

            if causal:
                # Query global index >= key global index is allowed.
                qi_idx = torch.arange(qs, qe).unsqueeze(1)
                kj_idx = torch.arange(ks, ke).unsqueeze(0)
                S_ij = S_ij.masked_fill(qi_idx < kj_idx, float("-inf"))

            # Online-softmax update (rescale the running stats and output).
            m_ij = S_ij.max(dim=-1).values           # tile row-max
            m_new = torch.maximum(m_i, m_ij)
            p = torch.exp(S_ij - m_new.unsqueeze(1))  # (bq, bk)
            alpha = torch.exp(m_i - m_new)            # rescale factor for old
            l_i = alpha * l_i + p.sum(dim=-1)
            O_i = alpha.unsqueeze(1) * O_i + p @ Vj
            m_i = m_new

        O[qs:qe] = O_i / l_i.unsqueeze(1)

    return O, peak_tile
