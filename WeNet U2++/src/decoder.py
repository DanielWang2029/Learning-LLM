"""Attention decoder(s) for the U2++ second pass (rescoring).

U2++ trains two attention decoders on top of the shared encoder: a left-to-right
(L2R) decoder and a right-to-left (R2L) decoder. Unlike the streaming CTC first
pass, these decoders are autoregressive over the *label* sequence and attend to
the whole encoder output, so they can use both left and right context. In the
second pass they are used only to **rescore** the CTC n-best (no autoregressive
search), so their scores can be combined (U2++, §3.1, §3.3).

Special ids used by the decoder vocabulary:
    content tokens 0..3  (A, B, C, D),  SOS = 4,  EOS = 5.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

SOS, EOS = 4, 5
DEC_VOCAB = 6  # 4 content tokens + SOS + EOS


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout,
                                               batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout,
                                                batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.SiLU(),
                                nn.Linear(d_ff, d_model))
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.n3 = nn.LayerNorm(d_model)

    def forward(self, x, memory, self_mask):
        h = self.n1(x)
        x = x + self.self_attn(h, h, h, attn_mask=self_mask, need_weights=False)[0]
        h = self.n2(x)
        x = x + self.cross_attn(h, memory, memory, need_weights=False)[0]
        x = x + self.ff(self.n3(x))
        return x


class AttentionDecoder(nn.Module):
    """A small Transformer decoder. ``reverse=True`` makes it right-to-left."""

    def __init__(self, d_model=64, num_heads=2, d_ff=128, num_layers=2,
                 dropout=0.1, reverse=False, max_len=64):
        super().__init__()
        self.reverse = reverse
        self.embed = nn.Embedding(DEC_VOCAB, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.layers = nn.ModuleList(
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, DEC_VOCAB)

    def _prep_targets(self, tokens: torch.Tensor) -> torch.Tensor:
        """content tokens (B, L) -> teacher-forcing (input, target) with SOS/EOS.

        For the R2L decoder the content sequence is reversed first, so the same
        left-to-right causal machinery scores it right-to-left.
        """
        if self.reverse:
            tokens = torch.flip(tokens, dims=[1])
        return tokens

    def forward(self, tokens, memory):
        """Return per-step log-probs for a batch of full target sequences.

        tokens: (B, L) content ids. Returns (in_ids, tgt_ids, logprobs) where
        logprobs is (B, L+1, vocab) aligned to predicting tgt_ids.
        """
        tokens = self._prep_targets(tokens)
        b, L = tokens.shape
        sos = torch.full((b, 1), SOS, dtype=torch.long, device=tokens.device)
        eos = torch.full((b, 1), EOS, dtype=torch.long, device=tokens.device)
        in_ids = torch.cat([sos, tokens], dim=1)          # (B, L+1)
        tgt_ids = torch.cat([tokens, eos], dim=1)         # (B, L+1)

        x = self.embed(in_ids) + self.pos[:, : in_ids.size(1)]
        S = in_ids.size(1)
        causal = torch.triu(torch.ones(S, S, device=tokens.device), diagonal=1).bool()
        for layer in self.layers:
            x = layer(x, memory, causal)
        logits = self.out(self.norm(x))
        return in_ids, tgt_ids, F.log_softmax(logits, dim=-1)

    @torch.no_grad()
    def score(self, tokens, memory) -> torch.Tensor:
        """Sum of per-token log-probs (incl. EOS) for each sequence. (B,)"""
        _, tgt_ids, logp = self.forward(tokens, memory)
        gathered = logp.gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)  # (B, L+1)
        return gathered.sum(dim=1)
