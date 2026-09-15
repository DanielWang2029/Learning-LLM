"""A long-context associative-recall ("needle") task.

Each example is a length-``T`` sequence of filler tokens. Somewhere in it sits a
MARK token immediately followed by a payload VALUE token. A QUERY token sits at
the very end. The model must read the QUERY position and output the payload value
— which means routing information from the (possibly distant) MARK to the end.

This is exactly the kind of long-range dependency that pure sliding-window (local)
attention struggles with, but that even a single global layer can resolve.
"""

from __future__ import annotations

import torch

# Token id layout.
PAD = 0
MARK = 1
QUERY = 2
N_SPECIAL = 3
N_FILLER = 10   # filler token types
N_VALUE = 10    # payload value classes (also the number of output classes)

FILLER_LO = N_SPECIAL
FILLER_HI = N_SPECIAL + N_FILLER
VALUE_LO = FILLER_HI
VALUE_HI = VALUE_LO + N_VALUE
VOCAB_SIZE = VALUE_HI


def make_batch(batch_size: int, seq_len: int, generator: torch.Generator):
    """Return (tokens, query_pos, target_value, payload_pos)."""
    tokens = torch.randint(FILLER_LO, FILLER_HI, (batch_size, seq_len),
                           generator=generator)
    # payload position uniformly in [1, seq_len-3] so distance-to-end varies widely
    payload_pos = torch.randint(1, seq_len - 2, (batch_size,), generator=generator)
    values = torch.randint(0, N_VALUE, (batch_size,), generator=generator)
    for i in range(batch_size):
        p = int(payload_pos[i])
        tokens[i, p] = MARK
        tokens[i, p + 1] = VALUE_LO + int(values[i])
    query_pos = torch.full((batch_size,), seq_len - 1, dtype=torch.long)
    tokens[torch.arange(batch_size), query_pos] = QUERY
    return tokens, query_pos, values, payload_pos


def describe_token(tok: int) -> str:
    if tok == PAD:
        return "PAD"
    if tok == MARK:
        return "MARK"
    if tok == QUERY:
        return "QUERY"
    if FILLER_LO <= tok < FILLER_HI:
        return f"f{tok - FILLER_LO}"
    return f"VAL={tok - VALUE_LO}"
