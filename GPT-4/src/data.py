"""Synthetic language with *learnable structure that rewards scale*.

To see a scaling law we need a task whose achievable loss keeps dropping as a
model gets bigger, down toward an irreducible floor. A high-order Markov
("n-gram") source is perfect: the next token depends on the previous ``order``
tokens through a fixed but rich conditional distribution. A tiny model can only
capture coarse statistics; a larger one captures more of the ``vocab**order``
contexts, so its cross-entropy approaches the source's true conditional
entropy. That irreducible entropy is the "floor" a scaling law extrapolates to.
"""

from __future__ import annotations

import numpy as np


class MarkovSource:
    """A fixed order-``k`` Markov chain over a small vocabulary.

    The transition table is drawn once from a Dirichlet distribution (sharpened
    so the process is predictable but non-trivial) and then frozen, so every
    model in the scaling family is trained and evaluated on the *same* language.
    """

    def __init__(self, vocab_size: int = 24, order: int = 2, seed: int = 0) -> None:
        self.vocab_size = vocab_size
        self.order = order
        rng = np.random.default_rng(seed)
        n_contexts = vocab_size**order
        # Sharp-ish Dirichlet -> each context has a few likely next tokens.
        self.table = rng.dirichlet(np.full(vocab_size, 0.3), size=n_contexts)
        self._pow = vocab_size ** np.arange(order)  # context hashing weights

    def _context_id(self, window: np.ndarray) -> np.ndarray:
        return (window * self._pow).sum(axis=-1)

    def sample(self, n_seq: int, seq_len: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        out = np.zeros((n_seq, seq_len), dtype=np.int64)
        out[:, : self.order] = rng.integers(0, self.vocab_size, size=(n_seq, self.order))
        cdf = np.cumsum(self.table, axis=1)
        for t in range(self.order, seq_len):
            window = out[:, t - self.order : t]
            ctx = self._context_id(window)
            u = rng.random(n_seq)[:, None]
            out[:, t] = (u > cdf[ctx]).sum(axis=1)
        return out

    def entropy_floor(self) -> float:
        """Stationary conditional entropy (nats) — the loss a perfect model reaches."""
        # Uniform weighting over contexts is a close approximation for a sharp,
        # well-mixed chain and keeps the demo dependency-free.
        p = np.clip(self.table, 1e-12, 1.0)
        h_per_context = -(p * np.log(p)).sum(axis=1)
        return float(h_per_context.mean())
