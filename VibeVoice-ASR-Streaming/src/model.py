"""A tiny causal Transformer that consumes interleaved speech + text.

This is the LLM backbone of VibeVoice-ASR-Streaming, in miniature. Following
§3.1, incoming speech frames and generated speaker-attributed text share a single
autoregressive context. Two input modalities are projected into one embedding
space:

  * audio frame -> Linear(n_mels -> d_model) + an "audio" type embedding,
  * text token  -> Embedding(vocab -> d_model) + a "text" type embedding.

A causal mask enforces that each token attends only to the past, so decoding a
chunk's text is conditioned on all previously observed speech and text — the
retained history that fixes speaker identities across the conversation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.h = n_heads
        self.dk = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, attn_mask):
        B, S, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, S, self.h, self.dk).transpose(1, 2)
        k = k.view(B, S, self.h, self.dk).transpose(1, 2)
        v = v.view(B, S, self.h, self.dk).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        scores = scores.masked_fill(attn_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, S, D)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x, attn_mask):
        x = x + self.attn(self.norm1(x), attn_mask)
        x = x + self.ff(self.norm2(x))
        return x


def sinusoidal(S, d_model, device):
    pos = torch.arange(S, device=device).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))
    pe = torch.zeros(S, d_model, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class StreamingSAASR(nn.Module):
    """Streaming speaker-attributed ASR LLM (small)."""

    def __init__(self, n_mels, vocab, d_model=96, n_layers=2, n_heads=4, d_ff=192):
        super().__init__()
        self.d_model = d_model
        self.audio_in = nn.Linear(n_mels, d_model)
        self.text_emb = nn.Embedding(vocab, d_model)
        self.type_emb = nn.Embedding(2, d_model)   # 0 = text, 1 = audio
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)

    def embed(self, audio, text_ids, is_audio):
        # audio: (B,S,n_mels)  text_ids: (B,S)  is_audio: (B,S) bool
        a = self.audio_in(audio)
        t = self.text_emb(text_ids)
        m = is_audio.unsqueeze(-1).float()
        x = m * a + (1 - m) * t
        x = x + self.type_emb(is_audio.long())
        x = x + sinusoidal(x.size(1), self.d_model, x.device).unsqueeze(0)
        return x

    def forward(self, audio, text_ids, is_audio, pad_mask=None):
        B, S = text_ids.shape
        x = self.embed(audio, text_ids, is_audio)
        causal = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
        attn_mask = causal.view(1, 1, S, S)
        if pad_mask is not None:  # pad_mask: (B,S) True where padded
            attn_mask = attn_mask | pad_mask.view(B, 1, 1, S)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        return self.head(self.norm(x))
