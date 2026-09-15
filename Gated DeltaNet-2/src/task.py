"""Key-value associative recall *with overwrite* — the demo task.

A sequence presents key/value bindings and then queries keys. Crucially, some
keys are WRITTEN TWICE (an overwrite), and the query must return the *latest*
value bound to the key. Returning a superposition of old and new values (what a
plain additive linear-attention state does) counts as wrong. This is exactly
the pressure point the paper targets: editing one association in a fixed-size
state without scrambling the rest.

Token layout (single integer vocabulary)::

    0                         PAD
    [1 .. n_vals]             VALUE tokens
    [.. n_keys]               WRITE-KEY tokens  (followed by their value token)
    [.. n_keys]               QUERY-KEY tokens  (target = current value)

A write event  (key j, value v)  ->  tokens [WKEY_j, VAL_v]
A query        (key j)           ->  token  [QKEY_j], with target VAL_{current}
Only query positions are scored (all other targets are the ignore index -100).
"""

from __future__ import annotations

import torch

IGNORE = -100


class RecallVocab:
    def __init__(self, n_keys: int = 6, n_vals: int = 10) -> None:
        self.n_keys = n_keys
        self.n_vals = n_vals
        self.PAD = 0
        self.val_base = 1                       # VAL_v = val_base + v
        self.wkey_base = 1 + n_vals             # WKEY_j = wkey_base + j
        self.qkey_base = 1 + n_vals + n_keys     # QKEY_j = qkey_base + j
        self.size = 1 + n_vals + 2 * n_keys

    def val_tok(self, v: int) -> int:
        return self.val_base + v

    def wkey_tok(self, j: int) -> int:
        return self.wkey_base + j

    def qkey_tok(self, j: int) -> int:
        return self.qkey_base + j


def make_example(vocab: RecallVocab, n_overwrite: int, rng: torch.Generator):
    """Build one sequence. Returns (tokens, targets, meta)."""
    n_keys, n_vals = vocab.n_keys, vocab.n_vals

    # Phase 1: write every key once with a random value.
    current = {}
    tokens, targets = [], []
    key_order = torch.randperm(n_keys, generator=rng).tolist()
    for j in key_order:
        v = int(torch.randint(n_vals, (1,), generator=rng))
        current[j] = v
        tokens += [vocab.wkey_tok(j), vocab.val_tok(v)]
        targets += [IGNORE, IGNORE]

    # Phase 2: overwrite a random subset of keys with new values.
    overwrite_keys = torch.randperm(n_keys, generator=rng).tolist()[:n_overwrite]
    for j in overwrite_keys:
        v = int(torch.randint(n_vals, (1,), generator=rng))
        current[j] = v                          # the *latest* binding
        tokens += [vocab.wkey_tok(j), vocab.val_tok(v)]
        targets += [IGNORE, IGNORE]

    # Phase 3: query every key; target is its current (latest) value.
    for j in torch.randperm(n_keys, generator=rng).tolist():
        tokens.append(vocab.qkey_tok(j))
        targets.append(vocab.val_tok(current[j]))

    meta = {"overwritten": sorted(overwrite_keys), "current": dict(current)}
    return tokens, targets, meta


def make_batch(
    vocab: RecallVocab,
    batch_size: int,
    n_overwrite: int,
    rng: torch.Generator,
):
    """Return (tokens, targets) tensors of shape (batch_size, seq_len)."""
    seqs, tgts = [], []
    for _ in range(batch_size):
        t, y, _ = make_example(vocab, n_overwrite, rng)
        seqs.append(t)
        tgts.append(y)
    return torch.tensor(seqs, dtype=torch.long), torch.tensor(tgts, dtype=torch.long)


def make_eval_batch(
    vocab: RecallVocab,
    batch_size: int,
    n_overwrite: int,
    rng: torch.Generator,
):
    """Like ``make_batch`` but also returns an ``overwritten`` boolean mask.

    ``overwritten[b, t]`` is True at query positions whose key was written more
    than once (i.e. the position where erasing an old value actually matters).
    """
    seqs, tgts, over = [], [], []
    for _ in range(batch_size):
        t, y, meta = make_example(vocab, n_overwrite, rng)
        ow_keys = set(meta["overwritten"])
        row = []
        for tok, tgt in zip(t, y):
            if tgt != IGNORE:  # a query position
                j = tok - vocab.qkey_base
                row.append(j in ow_keys)
            else:
                row.append(False)
        seqs.append(t)
        tgts.append(y)
        over.append(row)
    return (
        torch.tensor(seqs, dtype=torch.long),
        torch.tensor(tgts, dtype=torch.long),
        torch.tensor(over, dtype=torch.bool),
    )
