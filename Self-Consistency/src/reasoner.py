"""A controlled, stochastic chain-of-thought reasoner (no LLM required).

Self-consistency (Wang et al., 2022) works because a language model can reach
the *same correct answer through many different reasoning paths*. Greedy
decoding commits to the single most-probable path, which may be a confident but
wrong one; sampling many paths and marginalizing (majority vote) recovers the
answer that the largest share of *diverse* paths agree on.

Reproducing this faithfully needs a reasoner whose single most-likely path can
disagree with the answer most paths reach — i.e. ``argmax_path ≠ argmax_answer``.
Plain neural addition does not have this property (there is one canonical
scratchpad), so instead we use a transparent programmatic reasoner over a task
that genuinely admits many reasoning paths: **summing several numbers**, which
can be done in any order.

Task: add ``k`` numbers. The correct answer is reachable by every ordering
(many diverse paths). On some "trap" problems there is also a tempting shortcut
path (a systematic mistake) that has the *single highest* path-probability, so
GREEDY decoding follows it and is wrong — but the many correct orderings
together carry more probability, so SAMPLING + majority vote gets it right.

``temperature`` is the exact knob from the paper:
* ``T → 0`` (greedy) collapses onto the single most-likely path (the trap).
* ``T > 0`` spreads probability across the diverse correct paths.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field


@dataclass
class Problem:
    nums: list[int]
    answer: int
    trap: bool
    lure: int  # the systematically-wrong answer greedy is lured toward


@dataclass
class Path:
    logit: float           # unnormalized log-probability of choosing this path
    order: list[int] | None  # index order to add the numbers (None for the lure)
    kind: str              # "correct" or "lure"


# The gap that makes the shortcut the single most-probable path (but not the
# most-probable *answer*, since many correct orderings out-mass it together).
LURE_LOGIT = 1.3
CORRECT_LOGIT = 0.0
SLIP_PROB = 0.05  # per-addition chance of a small arithmetic slip when sampling


def generate_problems(n: int, k: int, seed: int, trap_frac: float = 0.4) -> list[Problem]:
    rng = random.Random(seed)
    out: list[Problem] = []
    for _ in range(n):
        nums = [rng.randint(10, 99) for _ in range(k)]
        answer = sum(nums)
        trap = rng.random() < trap_frac
        # A plausible systematic mistake: drop the smallest addend.
        lure = answer - min(nums) if trap else answer
        if lure == answer:  # keep the trap genuinely wrong
            trap = False
            lure = answer
        out.append(Problem(nums=nums, answer=answer, trap=trap, lure=lure))
    return out


def paths_for(problem: Problem) -> list[Path]:
    """All reasoning paths for a problem: every addition order, plus the lure."""
    k = len(problem.nums)
    paths = [
        Path(logit=CORRECT_LOGIT, order=list(order), kind="correct")
        for order in itertools.permutations(range(k))
    ]
    if problem.trap:
        paths.append(Path(logit=LURE_LOGIT, order=None, kind="lure"))
    return paths


def _softmax(logits: list[float], temperature: float) -> list[float]:
    t = max(temperature, 1e-6)
    m = max(logits)
    exps = [math.exp((x - m) / t) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


def _execute(problem: Problem, path: Path, rng: random.Random, allow_slip: bool):
    """Run one reasoning path -> (answer, human-readable trace steps)."""
    if path.kind == "lure":
        smallest = min(problem.nums)
        steps = [
            f"shortcut: skip the smallest addend ({smallest})",
            f"add the rest → {problem.lure}",
        ]
        return problem.lure, steps
    total = 0
    steps = []
    for j, idx in enumerate(path.order):
        v = problem.nums[idx]
        if allow_slip and rng.random() < SLIP_PROB:
            v += rng.choice([-2, -1, 1, 2, 10, -10])  # a small arithmetic slip
        total += v
        steps.append(f"{'start' if j == 0 else '+'} {problem.nums[idx]} → running total {total}")
    return total, steps


def greedy_answer(problem: Problem):
    """Deterministic greedy decode: follow the single highest-probability path."""
    paths = paths_for(problem)
    best = max(paths, key=lambda p: p.logit)
    ans, steps = _execute(problem, best, random.Random(0), allow_slip=False)
    return ans, steps, best.kind


def sample_answer(problem: Problem, rng: random.Random, temperature: float):
    """Sample one reasoning path (temperature) and execute it (with slips)."""
    paths = paths_for(problem)
    probs = _softmax([p.logit for p in paths], temperature)
    r = rng.random()
    cum = 0.0
    chosen = paths[-1]
    for p, pr in zip(paths, probs):
        cum += pr
        if r <= cum:
            chosen = p
            break
    ans, steps = _execute(problem, chosen, rng, allow_slip=True)
    return ans, steps, chosen.kind
