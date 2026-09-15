"""The toy multi-step reasoning task used to measure test-time scaling.

Gemini 2.5 is a *thinking* model whose accuracy improves when it is allowed to
spend more compute at test time. To measure that on a CPU we need a task where a
single reasoning attempt is fallible but more attempts help. We use multi-step
modular arithmetic: start from a value in ``0..9`` and apply a chain of
operations; the answer is the final running value (mod 10). A single noisy
reasoning pass may slip on a step; aggregating several passes recovers the truth.
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


def make_problems(n: int, length: int, generator: torch.Generator):
    """Sample ``n`` problems of a fixed chain ``length``."""
    problems = []
    for _ in range(n):
        start = int(torch.randint(0, MOD, (1,), generator=generator))
        op_ids = torch.randint(0, N_OPS, (length,), generator=generator).tolist()
        problems.append({"start": start, "op_ids": op_ids,
                         "answer": solve(start, op_ids), "length": length})
    return problems


def render(start: int, op_ids: list[int]) -> str:
    return f"{start} " + " ".join(OPS[o] for o in op_ids) + " = ?"
