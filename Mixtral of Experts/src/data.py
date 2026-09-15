"""A multi-domain toy language task with distinct sub-patterns.

To see experts *specialize*, the data must contain clearly separable structure.
We use ``N_DOMAINS`` domains, each with:

* its own disjoint block of ``DOMAIN_SIZE`` token ids, and
* its own arithmetic step, so within a domain the sequence advances by a
  domain-specific amount:  next = block_base + ((v + step_d) mod DOMAIN_SIZE).

A sequence lives entirely in one domain.  A good router learns to send each
domain's tokens to a consistent subset of experts — which is exactly the expert
specialization Mixtral reports (paper §5).
"""

from __future__ import annotations

import torch

N_DOMAINS = 4
DOMAIN_SIZE = 8
VOCAB_SIZE = N_DOMAINS * DOMAIN_SIZE      # 32
SEQ_LEN = 16
_STEPS = [1, 3, 5, 7]                     # per-domain advance (coprime with 8)


def _domain_sequence(domain: int, gen: torch.Generator) -> torch.Tensor:
    base = domain * DOMAIN_SIZE
    step = _STEPS[domain]
    v0 = int(torch.randint(0, DOMAIN_SIZE, (1,), generator=gen))
    vs = [(v0 + i * step) % DOMAIN_SIZE for i in range(SEQ_LEN)]
    return torch.tensor([base + v for v in vs], dtype=torch.long)


def make_dataset(n: int, seed: int):
    """Return (sequences, domain_labels): `n` sequences balanced across domains."""
    gen = torch.Generator().manual_seed(seed)
    seqs, labels = [], []
    for i in range(n):
        d = i % N_DOMAINS
        seqs.append(_domain_sequence(d, gen))
        labels.append(d)
    return torch.stack(seqs), torch.tensor(labels, dtype=torch.long)
