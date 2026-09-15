"""Topic-conditioned symbol mapping — a task that rewards routing capacity.

There are ``n_topics`` topics. Each topic ``g`` has its own fixed random
permutation ``perm_g`` of the ``n_symbols`` symbols. A single example is a
(symbol ``s``, topic ``g``) pair, and the target is ``perm_g(s)``.

Why this task? The symbol embedding is *shared* across topics, so predicting
the answer requires topic-dependent computation. One small dense network must
squeeze every topic's permutation into a single set of weights; a sparse MoE
can instead let different experts specialize on different topics while running
only a few experts per token. This mirrors the paper's claim that high total
capacity with few active parameters preserves quality.
"""

from __future__ import annotations

import torch


class TopicPermutations:
    def __init__(self, n_symbols: int, n_topics: int, seed: int = 0) -> None:
        self.n_symbols = n_symbols
        self.n_topics = n_topics
        g = torch.Generator().manual_seed(seed)
        # perms[g] is a random permutation of [0, n_symbols).
        self.perms = torch.stack(
            [torch.randperm(n_symbols, generator=g) for _ in range(n_topics)]
        )

    def batch(self, batch_size: int, rng: torch.Generator):
        topics = torch.randint(self.n_topics, (batch_size,), generator=rng)
        symbols = torch.randint(self.n_symbols, (batch_size,), generator=rng)
        targets = self.perms[topics, symbols]
        return symbols, topics, targets
