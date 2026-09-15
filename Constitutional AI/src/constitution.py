"""The constitution: explicit rules, critique, and revision.

This is the heart of Constitutional AI (Bai et al., 2022). A fixed set of
natural-language principles (the "constitution") is turned into checkers. Given
a response, the model:

  1. CRITIQUE — find the first constitutional principle it violates, and
  2. REVISE   — rewrite the response to comply with that principle,

iterating until the response is compliant. No human is in the loop; the
critique/revision are driven entirely by the written rules (here, deterministic
functions standing in for an LLM asked to self-critique against each rule).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from .words import MAX_LEN, REVISION_TOKEN, TOXIC_SET


@dataclass
class Principle:
    id: str
    text: str                                   # the constitutional principle
    violated: Callable[[List[str]], bool]       # does the response break it?
    revise: Callable[[List[str]], List[str]]    # rewrite to comply
    critique: str                               # what the critique step says


# -- individual checkers / revisers ------------------------------------- #
def _has_toxic(r: List[str]) -> bool:
    return any(w in TOXIC_SET for w in r)


def _revise_toxic(r: List[str]) -> List[str]:
    return [REVISION_TOKEN if w in TOXIC_SET else w for w in r]


def _has_repeat(r: List[str]) -> bool:
    return any(a == b for a, b in zip(r, r[1:]))


def _revise_repeat(r: List[str]) -> List[str]:
    out: List[str] = []
    for w in r:
        if not out or out[-1] != w:
            out.append(w)
    return out


def _too_long(r: List[str]) -> bool:
    return len(r) > MAX_LEN


def _revise_length(r: List[str]) -> List[str]:
    return r[:MAX_LEN]


# -- the constitution (checked top to bottom) --------------------------- #
CONSTITUTION: List[Principle] = [
    Principle(
        id="harmless",
        text="The response must not contain harmful or insulting language.",
        violated=_has_toxic, revise=_revise_toxic,
        critique="The response uses harmful/insulting words; remove them.",
    ),
    Principle(
        id="non-repetitive",
        text="The response must not repeat the same word twice in a row.",
        violated=_has_repeat, revise=_revise_repeat,
        critique="The response repeats a word; drop the duplication.",
    ),
    Principle(
        id="concise",
        text=f"The response must be at most {MAX_LEN} words.",
        violated=_too_long, revise=_revise_length,
        critique=f"The response is too long; trim it to {MAX_LEN} words.",
    ),
]


def first_violation(response: List[str]) -> Optional[Principle]:
    """The first constitutional principle the response violates, if any."""
    for p in CONSTITUTION:
        if p.violated(response):
            return p
    return None


def is_compliant(response: List[str]) -> bool:
    return first_violation(response) is None


def critique_and_revise(
    response: List[str], max_iters: int = 5
) -> Tuple[List[str], List[dict]]:
    """Run the critique → revise loop until compliant (or max_iters).

    Returns the final revised response and a trace of each critique/revision.
    """
    trace: List[dict] = []
    current = list(response)
    for _ in range(max_iters):
        p = first_violation(current)
        if p is None:
            break
        revised = p.revise(current)
        trace.append({
            "rule": p.id, "critique": p.critique,
            "before": list(current), "after": list(revised),
        })
        current = revised
    return current, trace


def violation_counts(responses: List[List[str]]) -> dict:
    """Per-principle violation counts plus overall compliance rate."""
    counts = {p.id: 0 for p in CONSTITUTION}
    compliant = 0
    for r in responses:
        for p in CONSTITUTION:
            if p.violated(r):
                counts[p.id] += 1
        if is_compliant(r):
            compliant += 1
    return {"per_rule": counts, "compliance_rate": compliant / len(responses)}
