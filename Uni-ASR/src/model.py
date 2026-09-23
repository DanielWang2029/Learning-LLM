"""Uni-ASR: one model for non-streaming AND streaming ASR (paper §2).

Architecture (§2.1): a **causal Conformer encoder**, an **adapter** (two linear
layers + ReLU), and an **LLM-style decoder**. Following the paper, speech and
text are arranged as an **interleaved sequence** and trained with **loss masks**
(§2.2.2, Fig. 2): after each speech segment comes its transcribed token, and the
loss is taken only on the text targets.

    [ a0 , t0 , a1 , t1 , ... , a_{k-1} , t_{k-1} ]
       └─ predict t_j at the position of speech segment a_j ─┘

Here each speech segment ``a_j`` is the adapter feature for token ``j``, pooled
over that token's encoder frames using forced alignment (the paper aligns text
to speech chunks with an external tool). Crucially the pool only covers frames
that have *already arrived* (``seen``), so a token whose acoustic nucleus has not
yet been streamed in gets an information-poor feature — the exact situation the
**latest-token fallback** (§2.3) is designed to repair: emit that last token
provisionally, then re-decode it once the next chunk delivers its nucleus, at no
added latency.

Because the encoder is causal, pooling ``[j·W : min((j+1)·W, seen)]`` from a
single full-clip encode is identical to encoding the stream chunk by chunk.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .conformer import ConformerEncoder, sinusoidal_positions

IGNORE = -100


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


class Adapter(nn.Module):
    """Two linear layers with a ReLU in between (paper §2.1)."""

    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_out), nn.ReLU(), nn.Linear(d_out, d_out)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderLM(nn.Module):
    """Decoder-only Transformer over a sequence of input embeddings."""

    def __init__(self, vocab_size, d_model, num_layers=2, num_heads=4, d_ff=192):
        super().__init__()
        self.d_model = d_model
        self.token_embed = nn.Embedding(vocab_size, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_ff, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, embeds: torch.Tensor) -> torch.Tensor:
        seq = embeds.size(1)
        pos = sinusoidal_positions(seq, self.d_model, embeds.device)
        x = embeds + pos.unsqueeze(0)
        causal = torch.triu(
            torch.ones(seq, seq, device=x.device, dtype=torch.bool), diagonal=1
        )
        x = self.transformer(x, mask=causal)
        return self.lm_head(self.norm(x))


class UniASR(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        frames_per_token: int,
        n_mels: int = 80,
        d_model: int = 96,
        d_llm: int = 96,
        enc_layers: int = 3,
        dec_layers: int = 2,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.W = frames_per_token
        self.encoder = ConformerEncoder(
            n_mels=n_mels, d_model=d_model, num_layers=enc_layers, num_heads=num_heads
        )
        self.adapter = Adapter(d_model, d_llm)
        self.decoder = DecoderLM(vocab_size, d_llm, dec_layers, num_heads)

    # ---- audio ----
    def encode(self, mel: torch.Tensor) -> torch.Tensor:
        """(B, n_mels, mel_frames) -> (B, T', d) causal per-frame embeddings."""
        return self.encoder(mel)

    def pool_tokens(self, frames: torch.Tensor, pool_ends: list[int]) -> torch.Tensor:
        """Adapter features per token, pooling forced-aligned frame windows.

        frames: (B, T', d).  pool_ends[j] = last frame (exclusive) available for
        token j; its window is [j*W : pool_ends[j]].  Returns (B, k, d_llm).
        """
        W = self.W
        feats = []
        for j, end in enumerate(pool_ends):
            lo = j * W
            hi = max(lo + 1, min(end, frames.size(1)))
            feats.append(frames[:, lo:hi].mean(dim=1))
        stacked = torch.stack(feats, dim=1)                # (B, k, d)
        return self.adapter(stacked)

    def _interleave(self, a_feats: torch.Tensor, tokens: list[int]) -> torch.Tensor:
        """Build [a0, t0, a1, t1, ...] ending in the next un-decoded a_j."""
        d = a_feats.size(-1)
        parts = []
        k = a_feats.size(1)
        for j in range(k):
            parts.append(a_feats[:, j : j + 1])
            if j < len(tokens):
                tok = self.decoder.token_embed(
                    torch.tensor([[tokens[j]]], device=a_feats.device)
                )
                parts.append(tok)
        return torch.cat(parts, dim=1)

    # ---- training: interleaved teacher forcing with loss masks (§2.2.2) ----
    def training_batch(self, frames: torch.Tensor, tokens: torch.Tensor,
                       seen: int, context_aware: bool):
        """Vectorized (embeds, targets) for a batch at truncation ``seen``.

        Interleaves speech/text and supervises only the text targets at each
        speech position. Context-aware (CS) masking drops the boundary token
        whose nucleus is not yet within ``seen`` (§2.3).
        """
        W = self.W
        B, N = tokens.shape
        k = min(N, ceil_div(seen, W))
        pool_ends = [min((j + 1) * W, seen) for j in range(k)]
        a = self.pool_tokens(frames, pool_ends)            # (B, k, d)
        tok_emb = self.decoder.token_embed(tokens[:, :k])  # (B, k, d)

        # interleave -> (B, 2k, d)
        seq = torch.stack([a, tok_emb], dim=2).reshape(B, 2 * k, a.size(-1))
        targets = torch.full((B, 2 * k), IGNORE, dtype=torch.long, device=frames.device)
        for j in range(k):
            if context_aware and (j + 1) * W > seen:
                continue                                    # CS: mask boundary token
            targets[:, 2 * j] = tokens[:, j]               # predict t_j at a_j
        return seq, targets

    # ---- streaming decode with forced-alignment chunking ----
    @torch.no_grad()
    def decode_stream(self, mel: torch.Tensor, N: int, chunk_frames: int,
                      fallback: bool) -> list[int]:
        """Chunked greedy decode. ``fallback`` re-pools and re-decodes the
        provisional last token when the next chunk arrives (§2.3)."""
        W = self.W
        frames = self.encode(mel)
        T = frames.size(1)
        committed: list[int] = []
        pool_ends: list[int] = []                          # frame budget used per token
        c = 0
        while True:
            seen = min((c + 1) * chunk_frames, T)
            target_count = min(N, ceil_div(seen, W))
            if fallback and c > 0 and committed:
                committed.pop()                            # drop provisional token
                pool_ends.pop()
            while len(committed) < target_count:
                j = len(committed)
                pool_ends.append(min((j + 1) * W, seen))   # forced-aligned window so far
                a = self.pool_tokens(frames, pool_ends)
                seq = self._interleave(a, committed)
                logits = self.decoder(seq)
                committed.append(int(logits[0, -1].argmax()))
            c += 1
            if seen >= T and target_count >= N:
                break
        return committed

    @torch.no_grad()
    def decode_full(self, mel: torch.Tensor, N: int) -> list[int]:
        """Non-streaming greedy decode from the complete audio."""
        frames = self.encode(mel)
        T = frames.size(1)
        committed: list[int] = []
        while len(committed) < N:
            j = len(committed)
            pool_ends = [min((i + 1) * self.W, T) for i in range(j + 1)]
            a = self.pool_tokens(frames, pool_ends)
            seq = self._interleave(a, committed)
            logits = self.decoder(seq)
            committed.append(int(logits[0, -1].argmax()))
        return committed
