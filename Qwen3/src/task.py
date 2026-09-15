"""The toy multi-step reasoning task (Qwen3 §"thinking" evaluation, at small scale).

Qwen3's headline behaviour is a single model that can *think* step by step
before answering. To make that measurable on a CPU we use a task whose answer
genuinely requires sequential computation: start from a value in ``0..9`` and
apply a chain of modular-arithmetic operations. The correct answer is the final
running value ``(mod 10)``.

A model that answers in one shot (non-thinking) has to collapse the whole chain
mentally; a model that *thinks* can apply the operations one at a time and carry
the running value forward — the essence of chain-of-thought / test-time compute.
"""

from __future__ import annotations

import torch

# Operation vocabulary. Each op maps a value ``v`` in 0..9 to a new value mod 10.
# Index ``len(OPS)`` (see reasoner) is reserved as a no-op / padding token.
OPS = ["+1", "+2", "+3", "+4", "-1", "-2", "-3", "*2", "*3"]
N_OPS = len(OPS)
MOD = 10  # answers live in 0..9


def apply_op(value: int, op_idx: int) -> int:
    """Apply one operation to a value (ground-truth transition, mod 10)."""
    op = OPS[op_idx]
    sign = op[0]
    k = int(op[1:])
    if sign == "+":
        value = value + k
    elif sign == "-":
        value = value - k
    else:  # "*"
        value = value * k
    return value % MOD


def solve(start: int, op_ids: list[int]) -> int:
    """Ground-truth answer: apply the whole chain left to right."""
    v = start
    for o in op_ids:
        v = apply_op(v, o)
    return v


def make_problems(n: int, min_len: int, max_len: int, generator: torch.Generator):
    """Sample ``n`` problems with chain length uniform in ``[min_len, max_len]``.

    Returns a list of dicts: ``{start, op_ids, answer, length}``.
    """
    problems = []
    for _ in range(n):
        length = int(torch.randint(min_len, max_len + 1, (1,), generator=generator))
        start = int(torch.randint(0, MOD, (1,), generator=generator))
        op_ids = torch.randint(0, N_OPS, (length,), generator=generator).tolist()
        problems.append(
            {"start": start, "op_ids": op_ids, "answer": solve(start, op_ids),
             "length": length}
        )
    return problems


def render(start: int, op_ids: list[int]) -> str:
    """Human-readable problem, e.g. ``3 +2 *3 -1 = ?``."""
    return f"{start} " + " ".join(OPS[o] for o in op_ids) + " = ?"


def render_think(start: int, op_ids: list[int]) -> str:
    """A transparent <think> trace showing the running value after each step."""
    v = start
    steps = [f"start={v}"]
    for o in op_ids:
        v = apply_op(v, o)
        steps.append(f"{OPS[o]}->{v}")
    return " ".join(steps)
