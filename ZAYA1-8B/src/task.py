"""A content-based retrieval task that genuinely needs attention.

Each sequence has exactly one *marked* token (a binary flag). The label is that
marked token's payload id. To solve it, the model must locate the marked
position by content and read off its payload — the canonical job of attention.

It also lets us probe compression: Compressed Convolutional Attention pools the
key/value sequence into ``L/r`` latent tokens, so the marked token's payload
must survive being convolved with its neighbors. The demo shows CCA preserves
retrieval quality while shrinking the KV-cache by ``r``.
"""

from __future__ import annotations

import torch


class MarkedRetrieval:
    def __init__(self, vocab_size: int, seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.seq_len = seq_len

    def batch(self, batch_size: int, rng: torch.Generator):
        L, V = self.seq_len, self.vocab_size
        tokens = torch.randint(V, (batch_size, L), generator=rng)
        marks = torch.zeros(batch_size, L, dtype=torch.long)
        pos = torch.randint(L, (batch_size,), generator=rng)
        marks[torch.arange(batch_size), pos] = 1
        labels = tokens[torch.arange(batch_size), pos]
        return tokens, marks, labels
