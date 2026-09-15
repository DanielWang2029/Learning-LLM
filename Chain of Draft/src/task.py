"""A multi-step arithmetic task rendered in three prompting *styles*.

Chain of Draft (CoD) is a *prompting strategy*: the same model is asked to solve
a problem three different ways, and we compare accuracy vs. token cost.

* **Standard / direct** (`S`) — answer only, no reasoning shown.
* **Chain of Thought** (`C`) — verbose natural-language step-by-step reasoning.
* **Chain of Draft** (`D`) — minimalistic drafts: just the essential
  intermediate result at each step, no prose.

The underlying task is a chained single-digit addition **modulo 10**
(``3+5+2+9+4`` → running sum mod 10). It is *multi-step*: the answer is the last
of several dependent sub-results. A small model struggles to compute it in one
shot (Standard), but if it is allowed to emit the running results step by step
(CoT or CoD) each step becomes a trivial ``(a+b) mod 10`` it can learn — so
reasoning *scaffolding* raises accuracy. CoD keeps only the numbers, so it
reaches the same accuracy as CoT with a small fraction of the tokens.

The prompt is ``<style><question>=``; the model generates the rest, ending with
``#<answer>`` and an end-of-sequence ``.``.
"""

from __future__ import annotations

import random
from typing import List, Tuple

# Style tags placed at the very start of the prompt.
STANDARD, COT, COD = "S", "C", "D"
STYLES = [STANDARD, COT, COD]
STYLE_NAME = {STANDARD: "Standard (direct)", COT: "Chain of Thought", COD: "Chain of Draft"}

# CoT uses a few English words; the vocabulary is the union of every character
# that can appear in any style. "." doubles as the end-of-sequence marker.
CHARS = list("0123456789+=#,. SCDaeghilmnoprstuw")
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
    ans = 0
    for t in terms:
        ans = (ans + t) % 10
    return terms, ans


def _question(terms: List[int]) -> str:
    return "+".join(str(t) for t in terms)


def _running(terms: List[int]) -> List[int]:
    r, out = 0, []
    for t in terms:
        r = (r + t) % 10
        out.append(r)
    return out


def _steps(terms: List[int]):
    """Yield (prev, term, result) for each running (prev+term) mod 10 step."""
    run = _running(terms)
    prev = terms[0]
    for t, r in zip(terms[1:], run[1:]):
        yield prev, t, r
        prev = r


def render_standard(terms: List[int]) -> str:
    """Answer only — no reasoning shown (the model must compute it in one shot)."""
    return f"#{_running(terms)[-1]}{EOS}"


def render_cot(terms: List[int]) -> str:
    """Verbose, natural-language step-by-step reasoning.

    Each step restates the operands in words, e.g. ``7 plus 7 is 4``, so the
    computation is a visible, self-contained ``(a+b) mod 10`` — but spelled out
    with many word tokens."""
    ans = _running(terms)[-1]
    clauses = [f"{prev} plus {t} is {r}" for prev, t, r in _steps(terms)]
    clauses.append(f"answer is {ans}")
    return ", ".join(clauses) + f"#{ans}{EOS}"


def render_cod(terms: List[int]) -> str:
    """Chain of Draft: the SAME steps as CoT, but as minimal symbolic drafts
    (``7+7=4 4+1=5 ...``) — no prose, only the essential equation per step."""
    ans = _running(terms)[-1]
    drafts = [f"{prev}+{t}={r}" for prev, t, r in _steps(terms)]
    return " ".join(drafts) + f"#{ans}{EOS}"


RENDER = {STANDARD: render_standard, COT: render_cot, COD: render_cod}


def prompt_of(style: str, terms: List[int]) -> str:
    """The prompt shown to the model (style tag + question up to '=')."""
    return f"{style}{_question(terms)}="


def full_example(style: str, terms: List[int]) -> str:
    return prompt_of(style, terms) + RENDER[style](terms)


def parse_answer(text: str):
    """Extract the integer answer following the last '#', if present."""
    if "#" not in text:
        return None
    tail = text.split("#")[-1]
    digits = ""
    for c in tail:
        if c.isdigit():
            digits += c
        else:
            break
    return int(digits) if digits else None
