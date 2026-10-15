"""Voxtral Realtime: causal encoder + adapter + DSM stream-synchronous decoder.

At each 80 ms frame the decoder performs one autoregressive step on a FUSED
input = adapter(audio_embedding_t) + embedding(previous_text_token), and predicts
the token for this frame ([P], [W], or a word). The decoder FFN is conditioned on
the target delay tau via AdaRMSNorm, so one model serves any delay.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import CausalSlidingWindowEncoder, RMSNorm, SwiGLU, _rope_tables, rope
from .adarmsnorm import AdaRMSNorm, DelayEmbedding


class DecoderBlock(nn.Module):
    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.n1 = RMSNorm(d)
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.ada = AdaRMSNorm(d)       # delay-conditioned norm on the FFN branch
        self.ff = SwiGLU(d, 4 * d)

    def _split(self, x):
        b, t, _ = x.shape
        return x.view(b, t, self.h, self.dh).transpose(1, 2)

    def forward(self, x, g_tau, cos, sin):
        b, t, d = x.shape
        h = self.n1(x)
        q, k, v = self._split(self.q(h)), self._split(self.k(h)), self._split(self.v(h))
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        scores = (q @ k.transpose(-2, -1)) / (self.dh ** 0.5)
        mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), 1)
        scores = scores.masked_fill(mask[None, None], float("-inf"))
        attn = (scores.softmax(-1) @ v).transpose(1, 2).reshape(b, t, d)
        x = x + self.o(attn)                                   # attention: unconditioned
        x = x + self.ff(self.ada(x, g_tau))                    # FFN: AdaRMSNorm(tau)
        return x


class VoxtralRealtime(nn.Module):
    def __init__(self, feat_dim: int, vocab: int, d: int = 64, enc_layers: int = 2,
                 dec_layers: int = 2, heads: int = 4, enc_window: int = 16,
                 max_len: int = 256) -> None:
        super().__init__()
        self.encoder = CausalSlidingWindowEncoder(feat_dim, d, enc_layers, heads, enc_window, max_len)
        self.adapter = nn.Linear(d, d)                         # (paper: + 4x temporal downsample)
        self.text_emb = nn.Embedding(vocab, d)
        self.delay = DelayEmbedding(d)
        self.blocks = nn.ModuleList(DecoderBlock(d, heads) for _ in range(dec_layers))
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, vocab)
        self.d, self.heads = d, heads

    def forward(self, feats, prev_tokens, tau) -> torch.Tensor:
        """feats (B,T,F), prev_tokens (B,T) ints, tau (B,) -> logits (B,T,vocab)."""
        enc = self.encoder(feats)
        x = self.adapter(enc) + self.text_emb(prev_tokens)     # DSM fusion (sum)
        g = self.delay(tau)
        cos, sin = _rope_tables(x.shape[1], self.d // self.heads, x.device)
        for blk in self.blocks:
            x = blk(x, g, cos, sin)
        return self.head(self.norm(x))
