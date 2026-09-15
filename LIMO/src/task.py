"""A chained-arithmetic reasoning task with two demonstration *qualities*.

LIMO argues that a *few high-quality* reasoning demonstrations elicit reasoning
better than *many low-quality* ones. To reproduce that contrast honestly we use
one task with two ways of *demonstrating* the solution — the final answer is
always correct in both; only the quality of the reasoning trace differs:

* **High quality** — a full step-by-step chain that exposes every intermediate
  result (running sum **modulo 10**):
  ``3+5+2+4=3+5=8;8+2=0;0+4=4#4``
* **Low quality** — a terse "shortcut" that jumps straight to the answer with no
  reasoning shown:  ``3+5+2+4=#4``

Both teach the same correct answer. The high-quality trace additionally teaches
the *process*: each step is a single-digit ``(a+b) mod 10`` lookup that is
**reused** across every step and every problem, so a small model can learn it
from only a handful of demonstrations and then generalize. The one-shot
function ``(Σ digits) mod 10`` is far harder to learn from answers alone — which
is exactly why *fewer high-quality* traces beat *many low-quality* ones here.

Grammar of a full example:
    <question> "=" <reasoning> "#" <answer> "."
where the model is prompted with everything up to and including "=", and must
generate the rest.
"""

from __future__ import annotations

import random
from typing import List, Tuple

# Fixed character vocabulary. "." doubles as the end-of-sequence marker.
CHARS = list("0123456789+=;#.")
EOS = "."


class Tokenizer:
    """Trivial character-level tokenizer over a fixed vocabulary."""

    def __init__(self) -> None:
        self.stoi = {c: i for i, c in enumerate(CHARS)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.eos_id = self.stoi[EOS]
        self.vocab_size = len(CHARS)

    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


def make_problem(num_terms: int, rng: random.Random) -> Tuple[List[int], int]:
    """Return ``(terms, answer)`` for a chained single-digit addition mod 10."""
    terms = [rng.randint(1, 9) for _ in range(num_terms)]
    return terms, sum(terms) % 10


def _question(terms: List[int]) -> str:
    return "+".join(str(t) for t in terms)


def high_quality_trace(terms: List[int]) -> str:
    """Full step-by-step reasoning: expose every (running-sum mod 10) step."""
    q = _question(terms)
    running = terms[0] % 10
    steps = []
    for t in terms[1:]:
        nxt = (running + t) % 10
        steps.append(f"{running}+{t}={nxt}")
        running = nxt
    reasoning = ";".join(steps)
    return f"{q}={reasoning}#{running}{EOS}"


def low_quality_trace(terms: List[int]) -> str:
    """Terse shortcut: correct answer, but no reasoning shown."""
    q = _question(terms)
    return f"{q}=#{sum(terms) % 10}{EOS}"


def prompt_of(terms: List[int]) -> str:
    """The prompt shown to the model at evaluation time (up to '=')."""
    return f"{_question(terms)}="


def build_dataset(
    n: int, num_terms: int, quality: str, seed: int
) -> List[str]:
    """Build ``n`` demonstration strings of the requested quality."""
    rng = random.Random(seed)
    make = high_quality_trace if quality == "high" else low_quality_trace
    seen = set()
    out = []
    # Sample distinct problems so counts reflect distinct demonstrations.
    while len(out) < n:
        terms, _ = make_problem(num_terms, rng)
        key = tuple(terms)
        if key in seen:
            continue
        seen.add(key)
        out.append(make(terms))
    return out
