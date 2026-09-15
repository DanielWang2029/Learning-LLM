"""The needle-in-a-haystack retrieval task (Gemini 1.5 long-context test).

Gemini 1.5's most striking long-context result is *near-perfect recall*: it can
retrieve a single planted fact ("needle") from a very long context ("haystack")
at essentially any position, up to millions of tokens.

We reproduce the structure of that test at tiny scale. Each example is a long run
of random distractor tokens (the haystack) with a single planted fact — a needle
marker followed by a secret value — inserted at some position, and a query at the
end asking for the value:

    d d d ... ⟨NEEDLE⟩ v ... d d d  ⟨QUERY⟩        -> the model must output  v

The model has to locate the unique ⟨NEEDLE⟩ marker anywhere in the haystack and
copy the token right after it. By sweeping the needle's position and the haystack
length we measure recall as a function of (position × context length) — the same
"needle recall" heatmap the paper reports.

Token layout over a fixed vocabulary:
    0 = PAD, 1 = ⟨NEEDLE⟩, 2 = ⟨QUERY⟩, values/distractors = [3 .. 3+N_TOKENS)
"""

from __future__ import annotations

import torch

PAD, NEEDLE, QUERY = 0, 1, 2
TOKEN_BASE = 3
N_TOKENS = 40                     # distinct distractor / value symbols
VOCAB_SIZE = TOKEN_BASE + N_TOKENS  # = 43


def _build(bs: int, n_fill: int, insert_at, gen: torch.Generator):
    """Assemble a batch given per-row needle insertion indices.

    ``insert_at``: (bs,) index in [0, n_fill] at which to splice the needle pair
    into the distractor stream.
    """
    fill = TOKEN_BASE + torch.randint(0, N_TOKENS, (bs, n_fill), generator=gen)
    values = TOKEN_BASE + torch.randint(0, N_TOKENS, (bs,), generator=gen)

    seqs = torch.empty(bs, n_fill + 3, dtype=torch.long)
    for r in range(bs):
        pos = int(insert_at[r])
        left, right = fill[r, :pos], fill[r, pos:]
        seqs[r] = torch.cat([left,
                             torch.tensor([NEEDLE, values[r]]),
                             right,
                             torch.tensor([QUERY])])
    return seqs, values


def make_batch(bs: int, n_fill: int, gen: torch.Generator):
    """Training batch with the needle at a uniformly random position."""
    insert_at = torch.randint(0, n_fill + 1, (bs,), generator=gen)
    return _build(bs, n_fill, insert_at, gen)


def make_eval_batch(bs: int, n_fill: int, needle_frac: float, gen: torch.Generator):
    """Evaluation batch with the needle at a fixed fractional depth in the haystack."""
    pos = min(n_fill, max(0, int(round(needle_frac * n_fill))))
    insert_at = torch.full((bs,), pos, dtype=torch.long)
    return _build(bs, n_fill, insert_at, gen)
