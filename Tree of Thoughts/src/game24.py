"""Game of 24 — the toy task from the Tree-of-Thoughts paper (Yao et al., 2023).

Given four numbers, reach 24 using +, -, ×, ÷ (each number used once). This is
the paper's headline example because a single left-to-right chain of thought
usually fails (it commits to early operations and cannot recover), while a
*search* over a tree of partial solutions succeeds.

State = the multiset of numbers still on the table. Each "thought" combines two
of them with one operator, shrinking the state by one number:

    (4 numbers) --op--> (3 numbers) --op--> (2 numbers) --op--> (1 number)

To keep everything integer-clean and CPU-trivial we only allow operations whose
result is an integer (division must be exact). Instances are filtered to those
solvable under this rule.
"""

from __future__ import annotations

from itertools import combinations

TARGET = 24


def moves(nums: tuple[int, ...]) -> list[tuple[tuple[int, ...], str]]:
    """All next states reachable by combining two numbers with one operator.

    Returns ``(new_state, description)`` pairs. Duplicate resulting states are
    de-duplicated (keeping the first operator that produces them) to keep the
    search tree readable.
    """
    def f(x: int) -> str:  # parenthesize negatives for readable traces
        return f"({x})" if x < 0 else str(x)

    out: list[tuple[tuple[int, ...], str]] = []
    seen: set[tuple[int, ...]] = set()
    n = len(nums)
    for i, j in combinations(range(n), 2):
        a, b = nums[i], nums[j]
        rest = [nums[k] for k in range(n) if k != i and k != j]
        candidates = [
            (a + b, f"{f(a)}+{f(b)}={a + b}"),
            (a * b, f"{f(a)}x{f(b)}={a * b}"),
            (a - b, f"{f(a)}-{f(b)}={a - b}"),
            (b - a, f"{f(b)}-{f(a)}={b - a}"),
        ]
        if b != 0 and a % b == 0:
            candidates.append((a // b, f"{f(a)}/{f(b)}={a // b}"))
        if a != 0 and b % a == 0:
            candidates.append((b // a, f"{f(b)}/{f(a)}={b // a}"))
        for val, desc in candidates:
            state = tuple(sorted(rest + [val]))
            if state in seen:
                continue
            seen.add(state)
            out.append((state, desc))
    return out


def can_reach_24(nums: tuple[int, ...]) -> bool:
    """Exact solver (integer ops only) — used to filter solvable instances."""
    if len(nums) == 1:
        return nums[0] == TARGET
    return any(can_reach_24(child) for child, _ in moves(nums))


def evaluate(nums: tuple[int, ...]) -> tuple[str, float]:
    """Heuristic state evaluator, in the spirit of the paper's sure/maybe/impossible.

    It is deliberately *imperfect* for states with 3+ numbers (a shallow,
    one-step lookahead), which is exactly why a greedy chain gets misled. For
    1- and 2-number states the judgement is exact and therefore a sound prune.

    Returns ``(label, value)`` where higher value = more promising.
    """
    if len(nums) == 1:
        return ("sure", 100.0) if nums[0] == TARGET else ("impossible", -100.0)
    if len(nums) == 2:
        reachable = any(child[0] == TARGET for child, _ in moves(nums))
        return ("sure", 50.0) if reachable else ("impossible", -100.0)
    # 3+ numbers: shallow lookahead — how close can one more op get us to 24?
    best = -1e9
    hits_target = False
    for child, _ in moves(nums):
        closest = min(abs(x - TARGET) for x in child)
        best = max(best, -closest)
        if TARGET in child:
            hits_target = True
    label = "sure" if hits_target else "maybe"
    return (label, best)
