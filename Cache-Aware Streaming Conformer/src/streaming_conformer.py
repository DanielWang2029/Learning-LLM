"""Cache-aware streaming Conformer (Noroozi et al. 2023).

The paper's central trick: process streaming audio chunk-by-chunk while
carrying two caches across chunks so that *no past frame is ever recomputed*
and the streaming output is **identical** to a full-sequence forward pass:

  * a **KV cache** — the self-attention keys/values of past frames, so the new
    chunk's queries can attend over history without re-projecting it (§3.3), and
  * a **conv cache** — the last ``K-1`` inputs to each causal depthwise
    convolution, so the convolution spans the chunk boundary correctly (§3.3).

Everything that has no temporal context (feed-forward, layer norm, pointwise
1×1 convolutions) needs no cache. To make this consistent, all convolutions are
**causal** (left-padded) and BatchNorm is replaced by LayerNorm (§3.1).

Attention uses **chunk-aware look-ahead**: every query in a chunk may attend to
all frames up to the end of its own chunk (plus a fixed left context), which is
what a single model needs to expose a latency/accuracy trade-off just by
changing the chunk size at inference time.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_chunk_mask(T: int, chunk_size: int, left_context: Optional[int],
                     device=None) -> torch.Tensor:
    """Chunk-aware attention mask, shape (T, T), True where attending is allowed.

    Query ``i`` (in chunk ``c = i // chunk_size``) may attend to key ``j`` iff
    ``c*chunk_size - left_context <= j <= c*chunk_size + chunk_size - 1``.
    ``left_context=None`` means unlimited left context.
    """
    i = torch.arange(T, device=device).unsqueeze(1)
    j = torch.arange(T, device=device).unsqueeze(0)
    chunk_start = (i // chunk_size) * chunk_size
    upper = chunk_start + chunk_size - 1
    allowed = j <= upper
    if left_context is not None:
        allowed &= j >= (chunk_start - left_context)
    return allowed


class StreamingAttention(nn.Module):
    """Multi-head self-attention with a KV cache for chunked streaming."""

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.h = num_heads
        self.d_k = d_model // num_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.h, self.d_k).transpose(1, 2)  # (b,h,t,dk)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Full-sequence forward with a (T, T) boolean allow-mask."""
        q, k, v = self._heads(self.q(x)), self._heads(self.k(x)), self._heads(self.v(x))
        scores = (q @ k.transpose(-2, -1)) / self.d_k ** 0.5
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))
        ctx = (attn @ v).transpose(1, 2).reshape(x.size(0), x.size(1), -1)
        return self.out(ctx)

    def forward_chunk(self, x: torch.Tensor, cache, left_context):
        """Streaming forward for one chunk.

        ``cache`` is a dict with cached keys/values (or None). The chunk's
        queries attend over [cached history + current chunk] with no mask —
        which exactly reproduces the chunk-aware allowed region.
        """
        q = self._heads(self.q(x))
        k = self._heads(self.k(x))
        v = self._heads(self.v(x))
        if cache is not None:
            k = torch.cat([cache["k"], k], dim=2)
            v = torch.cat([cache["v"], v], dim=2)
        scores = (q @ k.transpose(-2, -1)) / self.d_k ** 0.5
        attn = F.softmax(scores, dim=-1)  # eval-time: dropout is a no-op
        ctx = (attn @ v).transpose(1, 2).reshape(x.size(0), x.size(1), -1)
        # Keep only the frames a future chunk is still allowed to see.
        if left_context is not None:
            k = k[:, :, -left_context:, :]
            v = v[:, :, -left_context:, :]
        new_cache = {"k": k, "v": v}
        return self.out(ctx), new_cache, k.size(2)


class CausalConvModule(nn.Module):
    """Conformer conv module made causal, with a conv-state cache (§3.1/§3.3)."""

    def __init__(self, d_model: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.pointwise1 = nn.Conv1d(d_model, 2 * d_model, 1)
        self.depthwise = nn.Conv1d(d_model, d_model, kernel_size,
                                   padding=0, groups=d_model)
        self.norm = nn.LayerNorm(d_model)   # LayerNorm, not BatchNorm (streaming-safe)
        self.pointwise2 = nn.Conv1d(d_model, d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def _post(self, y: torch.Tensor) -> torch.Tensor:
        # y: (B, d, L). LayerNorm over channels (per frame), Swish, pointwise.
        y = self.norm(y.transpose(1, 2)).transpose(1, 2)
        y = self.pointwise2(F.silu(y))
        return self.dropout(y.transpose(1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = F.glu(self.pointwise1(x.transpose(1, 2)), dim=1)   # (B, d, T)
        g = F.pad(g, (self.kernel_size - 1, 0))                 # causal left pad
        return self._post(self.depthwise(g))

    def forward_chunk(self, x: torch.Tensor, cache):
        g = F.glu(self.pointwise1(x.transpose(1, 2)), dim=1)   # (B, d, C)
        if cache is None:
            pad = self.kernel_size - 1
            cache = torch.zeros(g.size(0), g.size(1), pad, device=g.device)
        g_cat = torch.cat([cache, g], dim=2)                   # prepend history
        y = self.depthwise(g_cat)                              # valid conv -> len C
        new_cache = g_cat[:, :, -(self.kernel_size - 1):]
        return self._post(y), new_cache


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class StreamingConformerBlock(nn.Module):
    """Conformer block with cache-aware streaming support."""

    def __init__(self, d_model, num_heads, d_ff, kernel_size, dropout):
        super().__init__()
        self.ffn1 = FeedForward(d_model, d_ff, dropout)
        self.attn = StreamingAttention(d_model, num_heads, dropout)
        self.conv = CausalConvModule(d_model, kernel_size, dropout)
        self.ffn2 = FeedForward(d_model, d_ff, dropout)
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_conv = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x = x + 0.5 * self.ffn1(x)
        x = x + self.attn(self.norm_attn(x), mask)
        x = x + self.conv(self.norm_conv(x))
        x = x + 0.5 * self.ffn2(x)
        return self.norm_out(x)

    def forward_chunk(self, x, cache, left_context):
        x = x + 0.5 * self.ffn1(x)
        a, kv, n_keys = self.attn.forward_chunk(self.norm_attn(x), cache["kv"], left_context)
        x = x + a
        c, conv = self.conv.forward_chunk(self.norm_conv(x), cache["conv"])
        x = x + c
        x = x + 0.5 * self.ffn2(x)
        return self.norm_out(x), {"kv": kv, "conv": conv}, n_keys


class StreamingConformerEncoder(nn.Module):
    """A small cache-aware streaming Conformer for frame classification.

    ``forward`` runs the whole sequence at once (used in training and as the
    reference for the equality check). ``forward_streaming`` processes the same
    input chunk-by-chunk with KV + conv caches; the two must agree numerically.
    """

    def __init__(self, n_mels=40, d_model=64, num_heads=2, d_ff=128,
                 num_blocks=2, kernel_size=9, num_classes=4, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(n_mels, d_model)
        self.blocks = nn.ModuleList(
            StreamingConformerBlock(d_model, num_heads, d_ff, kernel_size, dropout)
            for _ in range(num_blocks)
        )
        self.classifier = nn.Linear(d_model, num_classes)
        self.kernel_size = kernel_size

    def forward(self, feats, chunk_size, left_context=None):
        x = self.proj(feats)
        mask = build_chunk_mask(x.size(1), chunk_size, left_context, x.device)
        for blk in self.blocks:
            x = blk(x, mask)
        return self.classifier(x)

    @torch.no_grad()
    def forward_streaming(self, feats, chunk_size, left_context=None):
        """Chunk-by-chunk inference with caches. Returns (logits, key_counts).

        ``key_counts[c]`` is how many key frames the last block attended over
        while processing chunk ``c`` — this stays constant once the KV cache is
        full, evidence that there is no recomputation of past frames.
        """
        x = self.proj(feats)
        T = x.size(1)
        caches = [{"kv": None, "conv": None} for _ in self.blocks]
        chunk_outputs, key_counts = [], []
        for start in range(0, T, chunk_size):
            xc = x[:, start : start + chunk_size]
            last_keys = 0
            for i, blk in enumerate(self.blocks):
                xc, caches[i], n_keys = blk.forward_chunk(xc, caches[i], left_context)
                last_keys = n_keys
            chunk_outputs.append(self.classifier(xc))
            key_counts.append(last_keys)
        return torch.cat(chunk_outputs, dim=1), key_counts
