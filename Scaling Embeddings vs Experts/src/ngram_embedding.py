"""N-gram Embedding (a.k.a. Over-Encoding) — the "scaling embeddings" dimension.

Paper §2, Eq. 1-3. The embedding of token t_i is augmented with hashed n-gram
sub-table lookups over the local context:

    e_i = (1 / ((N-1)K + 1)) * [ E0(t_i)
              + Σ_{n=2}^N Σ_{k=1}^K  W_{n,k} · E_{n,k}( H_{n,k}(t_{i-n+1..i}) ) ]

    H_n(t_{i-n+1..i}) = ( Σ_{j=0}^{n-1} t_{i-j} · V0^j ) mod V_n     (rolling hash)

Each n-gram sub-table ``E_{n,k}`` has its own (large) vocabulary ``V_n`` but is
addressed by a *hash* of the n-gram, so it adds parameters without needing an
explicit n-gram vocabulary. This "densifies information per token": the bigram
/ trigram context is baked directly into the input representation.

This is a compact, faithful implementation with K=1 sub-table per order and an
optional linear projection W per order.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class NGramEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        orders: tuple = (2, 3),
        hash_vocab: int = 4096,
        project: bool = False,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.orders = tuple(orders)
        self.hash_vocab = hash_vocab

        self.base = nn.Embedding(vocab_size, d_model)      # E0
        # One hashed sub-table per requested n-gram order.
        self.ngram_tables = nn.ModuleList(
            [nn.Embedding(hash_vocab, d_model) for _ in self.orders]
        )
        if project:
            self.proj = nn.ModuleList(
                [nn.Linear(d_model, d_model, bias=False) for _ in self.orders]
            )
        else:
            self.proj = None
        self.scale = 1.0 / (len(self.orders) + 1)          # (N-1)K+1 with K=1

    def _hash(self, tokens: torch.Tensor, n: int) -> torch.Tensor:
        """Polynomial rolling hash of the length-n context ending at each pos.

        tokens: (B, T). Returns (B, T) hash ids in [0, hash_vocab). Positions
        without a full n-gram (t < n-1) fall back to a stable partial hash.
        """
        B, T = tokens.shape
        V0 = self.vocab_size
        acc = torch.zeros(B, T, dtype=torch.long, device=tokens.device)
        for j in range(n):                                 # t_{i-j} · V0^j
            shifted = torch.zeros_like(tokens)
            if j == 0:
                shifted = tokens
            else:
                shifted[:, j:] = tokens[:, :-j]
            acc = acc + shifted * (V0 ** j)
        return acc % self.hash_vocab

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        e = self.base(tokens)
        for idx, table in enumerate(self.ngram_tables):
            n = self.orders[idx]
            h = self._hash(tokens, n)
            emb = table(h)
            if self.proj is not None:
                emb = self.proj[idx](emb)
            e = e + emb
        return e * self.scale
