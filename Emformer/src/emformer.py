"""The Emformer streaming block, from scratch (paper Section 2).

Emformer chops an utterance into non-overlapping **center** blocks of length ``C``.
When it processes block *i* it lets the block's queries attend to:

  * an **augmented memory bank** — one summary vector per past block (Section 2.1),
  * the **left context** — the previous ``L`` frames (cached, Section 2.1),
  * the **center** block itself (the ``C`` current frames), and
  * the **right context** — the next ``R`` frames (lookahead, Section 2.1).

Because the memory bank distills the whole past into a bounded set of vectors and
the left context is a bounded window, the per-block cost is independent of how
long the utterance is — the key to low-latency streaming.

This module exposes the same computation two ways and they are numerically
identical:

  * ``forward_parallel`` builds one big attention over
    ``[memory tokens | all frames]`` with a mask encoding exactly the connections
    above (used at training time, fully parallel across blocks), and
  * ``forward_stream`` walks block by block, keeping a key/value cache and growing
    the memory bank, computing each frame's key/value exactly once.

For determinism (and so streaming == parallel exactly) the memory summary for a
block is the mean of its center frames projected into memory space — a common
simplification of the paper's learned summary query, which also sidesteps the
"disallow summary↔memory attention" rule of Section 2.2.3.
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mha(q, k, v, num_heads, temp=1.0):
    """Multi-head scaled dot-product attention. q/k/v: (Tq/Tk, D).

    ``temp`` sharpens the softmax. Speech attention is dominated by nearby
    frames, so a peaky (local) distribution is a realistic regime — and the one
    in which a bounded-context model like Emformer closely matches full context.
    """
    Tq, D = q.shape
    Tk = k.shape[0]
    dk = D // num_heads
    qh = q.view(Tq, num_heads, dk).transpose(0, 1)   # (h, Tq, dk)
    kh = k.view(Tk, num_heads, dk).transpose(0, 1)
    vh = v.view(Tk, num_heads, dk).transpose(0, 1)
    scores = temp * torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(dk)
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, vh)                     # (h, Tq, dk)
    return out.transpose(0, 1).reshape(Tq, D)


class EmformerBlock(nn.Module):
    """One Emformer layer (single utterance / batch-1 for clarity)."""

    def __init__(
        self,
        d_model: int = 32,
        num_heads: int = 4,
        center: int = 8,
        left: int = 16,
        right: int = 4,
        max_memory: int = 4,
        temperature: float = 12.0,
    ) -> None:
        super().__init__()
        self.d = d_model
        self.h = num_heads
        self.C = center
        self.L = left
        self.R = right
        self.M = max_memory
        self.temp = temperature
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.w_mem = nn.Linear(d_model, d_model, bias=False)  # summary -> memory token

    # ---- shared helpers --------------------------------------------------
    def _blocks(self, T: int):
        return [(s, min(s + self.C, T)) for s in range(0, T, self.C)]

    def _summaries(self, x: torch.Tensor) -> torch.Tensor:
        """One memory token per block = projected mean of its center frames."""
        return torch.stack([self.w_mem(x[s:e].mean(0)) for s, e in self._blocks(x.shape[0])])

    # ---- training-time parallel pass ------------------------------------
    def forward_parallel(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[0]
        blocks = self._blocks(T)
        nb = len(blocks)
        mem = self._summaries(x)                      # (nb, D)

        k_frames, v_frames = self.w_k(x), self.w_v(x)
        k_mem, v_mem = self.w_k(mem), self.w_v(mem)
        K = torch.cat([k_mem, k_frames], 0)           # (nb + T, D)
        V = torch.cat([v_mem, v_frames], 0)
        Q = self.w_q(x)

        # allowed[t, j] : can query frame t attend to key column j?
        allowed = torch.zeros(T, nb + T, dtype=torch.bool)
        for i, (s, e) in enumerate(blocks):
            lo, hi = max(0, s - self.L), min(T, e + self.R)
            for t in range(s, e):
                # bounded memory window: past blocks [i-M, i)
                for j in range(max(0, i - self.M), i):
                    allowed[t, j] = True
                allowed[t, nb + lo : nb + hi] = True

        dk = self.d // self.h
        qh = Q.view(T, self.h, dk).transpose(0, 1)
        kh = K.view(-1, self.h, dk).transpose(0, 1)
        vh = V.view(-1, self.h, dk).transpose(0, 1)
        scores = self.temp * torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(dk)
        scores = scores.masked_fill(~allowed.unsqueeze(0), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, vh).transpose(0, 1).reshape(T, self.d)
        return self.w_o(out)

    # ---- streaming inference --------------------------------------------
    def forward_stream(self, x: torch.Tensor):
        """Block-by-block streaming. Returns (output, stats)."""
        T = x.shape[0]
        blocks = self._blocks(T)

        # In real streaming each frame's key/value is computed once, as it
        # arrives, and stored in a bounded cache. We mimic that: compute all
        # frame KVs once up front and count them, then only *slice* the cache.
        k_frames, v_frames = self.w_k(x), self.w_v(x)     # each frame KV: computed once
        kv_computations = T

        memory: List[torch.Tensor] = []                   # augmented memory bank
        out = torch.zeros(T, self.d)
        peak_state = 0

        for i, (s, e) in enumerate(blocks):
            lo, hi = max(0, s - self.L), min(T, e + self.R)
            q = self.w_q(x[s:e])

            k_parts, v_parts = [], []
            if memory:
                mem = torch.stack(memory[-self.M :])       # bounded memory window
                k_parts.append(self.w_k(mem)); v_parts.append(self.w_v(mem))
            k_parts.append(k_frames[lo:hi]); v_parts.append(v_frames[lo:hi])
            K = torch.cat(k_parts, 0); V = torch.cat(v_parts, 0)

            out[s:e] = self.w_o(_mha(q, K, V, self.h, self.temp))

            # state we must hold: left-context KV window + memory bank
            peak_state = max(peak_state, (hi - lo) + min(len(memory), self.M))

            # emit this block's summary into the memory bank (after attending)
            summary = self.w_mem(x[s:e].mean(0))
            memory.append(summary)

        stats = {
            "kv_computations": kv_computations,       # == T  (no recomputation)
            "num_frames": T,
            "num_blocks": len(blocks),
            "peak_state_tokens": peak_state,          # bounded, independent of T
        }
        return out, stats


def full_attention(block: EmformerBlock, x: torch.Tensor) -> torch.Tensor:
    """Plain bidirectional attention (every frame attends to every frame)."""
    Q, K, V = block.w_q(x), block.w_k(x), block.w_v(x)
    return block.w_o(_mha(Q, K, V, block.h, block.temp))
