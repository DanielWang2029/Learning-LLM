"""A trigram-grammar language task with strong local n-gram structure.

A fixed random transition table ``T[a, b, c] -> next`` defines a deterministic
"grammar": each token is (usually) determined by the *previous three* tokens.
We inject a random token with small probability to keep the trigram contexts
diverse and the task non-trivial.

Because the next token depends on the (a, b, c) *trigram*, an N-gram embedding
that hashes that trigram directly into the input representation captures the
predictive signal in a single lookup — exactly the "densify information per
token" effect the paper studies. A 1-layer attention + MoE model *can* also
gather the three previous tokens, but composing their 3-way interaction is
expensive; the question is which allocation of a FIXED parameter budget learns
the grammar better.
"""

from __future__ import annotations

import torch


class TrigramGrammar:
    def __init__(self, vocab_size: int, seed: int = 0) -> None:
        self.vocab_size = vocab_size
        g = torch.Generator().manual_seed(seed)
        # T[a, b, c] -> next token (a fixed random function of the trigram).
        self.table = torch.randint(
            vocab_size, (vocab_size, vocab_size, vocab_size), generator=g
        )

    IGNORE = -100

    def batch(self, batch_size: int, seq_len: int, rng: torch.Generator,
              noise: float = 0.15):
        """Return (inputs, targets) of shape (batch, seq_len-1).

        ``targets[:, j]`` is the grammar-correct next token given the trigram
        (token j-2, j-1, j); it is the ignore index at j<2 (no trigram yet).
        Noise diversifies the inputs but the label is always the deterministic
        grammar output, so a perfect model can reach ~100% accuracy.
        """
        V = self.vocab_size
        seqs = torch.zeros(batch_size, seq_len, dtype=torch.long)
        seqs[:, :3] = torch.randint(V, (batch_size, 3), generator=rng)
        for i in range(3, seq_len):
            a, b, c = seqs[:, i - 3], seqs[:, i - 2], seqs[:, i - 1]
            nxt = self.table[a, b, c]
            rand = torch.randint(V, (batch_size,), generator=rng)
            use_rand = torch.rand(batch_size, generator=rng) < noise
            seqs[:, i] = torch.where(use_rand, rand, nxt)

        inputs = seqs[:, :-1]                              # (B, L-1)
        targets = torch.full_like(inputs, self.IGNORE)
        a = seqs[:, :-3]                                   # tokens 0..L-4
        b = seqs[:, 1:-2]
        c = seqs[:, 2:-1]
        targets[:, 2:] = self.table[a, b, c]               # label at j>=2
        return inputs, targets
