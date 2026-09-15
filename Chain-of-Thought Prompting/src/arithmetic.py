"""Multi-digit addition as a language task, in two answer formats.

This is the toy problem that stands in for the paper's reasoning benchmarks.
Each example is a string the language model must complete after the prompt
``"a+b="``:

* ``direct`` — jump straight to the answer, most-significant digit first::

      12+345=357$

  This is genuinely hard for a small model: carries propagate right-to-left,
  but the answer is emitted left-to-right, so the model must "look ahead".

* ``cot``  — a chain-of-thought scratchpad that adds digit by digit from the
  least-significant end, carrying explicitly, and only then states the answer::

      12+345=2+5+0=7c0|1+4+0=5c0|0+3+0=3c0#357$

  Every scratchpad step is a *local* computation (two digits + a carry -> one
  digit + a carry), which is exactly the decomposition chain-of-thought
  prompting encourages. The final answer after ``#`` is the same number.

The demo trains one model per format and compares exact-match accuracy.
"""

from __future__ import annotations

import random

STEP_SEP = "|"
ANS_SEP = "#"
EOS = "$"


def cot_scratchpad(a: int, b: int) -> str:
    """Return the digit-by-digit, carry-explicit reasoning trace for ``a+b``."""
    da, db = str(a), str(b)
    n = max(len(da), len(db))
    da, db = da.zfill(n), db.zfill(n)
    steps = []
    carry = 0
    # Work from the least-significant digit, like grade-school addition.
    for i in range(n - 1, -1, -1):
        x, y = int(da[i]), int(db[i])
        total = x + y + carry
        digit, carry_out = total % 10, total // 10
        # "x+y+carry_in=digit c carry_out"
        steps.append(f"{x}+{y}+{carry}={digit}c{carry_out}")
        carry = carry_out
    return STEP_SEP.join(steps)


def format_example(a: int, b: int, mode: str) -> tuple[str, str]:
    """Return ``(prompt, target)`` strings for the given ``mode``."""
    prompt = f"{a}+{b}="
    answer = str(a + b)
    if mode == "direct":
        target = f"{answer}{EOS}"
    elif mode == "cot":
        target = f"{cot_scratchpad(a, b)}{ANS_SEP}{answer}{EOS}"
    else:  # pragma: no cover
        raise ValueError(f"unknown mode: {mode}")
    return prompt, target


def parse_answer(text: str, mode: str) -> str | None:
    """Extract the predicted answer digits from a generated completion."""
    text = text.split(EOS)[0]  # drop anything after end-of-sequence
    if mode == "cot":
        if ANS_SEP not in text:
            return None
        text = text.split(ANS_SEP)[-1]
    return text if text.isdigit() else None


def sample_problem(rng: random.Random, min_digits: int, max_digits: int) -> tuple[int, int]:
    """Sample two operands whose digit-lengths lie in ``[min_digits, max_digits]``."""

    def one() -> int:
        d = rng.randint(min_digits, max_digits)
        lo = 10 ** (d - 1) if d > 1 else 0
        hi = 10 ** d - 1
        return rng.randint(lo, hi)

    return one(), one()


def make_dataset(
    n: int,
    min_digits: int,
    max_digits: int,
    seed: int,
) -> list[tuple[int, int]]:
    """Deterministically sample ``n`` unique addition problems."""
    rng = random.Random(seed)
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    while len(out) < n:
        a, b = sample_problem(rng, min_digits, max_digits)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        out.append((a, b))
    return out
