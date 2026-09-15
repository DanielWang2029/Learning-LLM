"""The toy multi-step task + the tiny curated SFT set (s1, 2025).

s1 makes two moves: (1) a *tiny curated* supervised set (~1K examples) is enough
to teach a reasoning format, and (2) **budget forcing** controls test-time compute
by appending "Wait" to keep the model thinking (or an end token to stop it).

Our task is multi-step modular arithmetic: start from a value in ``0..9`` and
apply a chain of operations; the answer is the final running value (mod 10). The
curated SFT set is deliberately skewed toward *short* chains, so a model trained
on it learns to stop thinking early — which is exactly what budget forcing then
corrects at test time.
"""

from __future__ import annotations

import torch

OPS = ["+1", "+2", "+3", "+4", "-1", "-2", "-3", "*2", "*3"]
N_OPS = len(OPS)
MOD = 10


def apply_op(value: int, op_idx: int) -> int:
    op = OPS[op_idx]
    sign, k = op[0], int(op[1:])
    if sign == "+":
        value += k
    elif sign == "-":
        value -= k
    else:
        value *= k
    return value % MOD


def solve(start: int, op_ids: list[int]) -> int:
    v = start
    for o in op_ids:
        v = apply_op(v, o)
    return v


def sample_length(max_len: int, generator: torch.Generator, skew: bool) -> int:
    """Sample a chain length. ``skew=True`` biases toward *short* chains (the
    curated SFT distribution); ``skew=False`` is uniform (the test distribution)."""
    if skew:
        # Poisson(lambda=2)-shaped lengths: short chains dominate and the per-step
        # "stop" hazard rises with length, so the SFT'd model learns to stop
        # thinking around step ~3 and under-thinks on longer problems.
        lengths = torch.arange(1, max_len + 1, dtype=torch.float)
        lam = 2.0
        weights = torch.exp(lengths * torch.log(torch.tensor(lam))
                            - torch.lgamma(lengths + 1))
        return int(torch.multinomial(weights, 1, generator=generator)) + 1
    return int(torch.randint(1, max_len + 1, (1,), generator=generator))


def make_problems(n: int, max_len: int, generator: torch.Generator, skew: bool):
    problems = []
    for _ in range(n):
        length = sample_length(max_len, generator, skew)
        start = int(torch.randint(0, MOD, (1,), generator=generator))
        op_ids = torch.randint(0, N_OPS, (length,), generator=generator).tolist()
        problems.append({"start": start, "op_ids": op_ids,
                         "answer": solve(start, op_ids), "length": length})
    return problems


def render(start: int, op_ids: list[int]) -> str:
    return f"{start} " + " ".join(OPS[o] for o in op_ids) + " = ?"
