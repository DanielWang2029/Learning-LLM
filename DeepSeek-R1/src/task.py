"""The toy task, the character vocabulary, and the reward function.

Task: given ``a+b=`` (with ``a, b`` small), emit a response of the form

    {a+b}[s]

where the ``{...}`` block is the *reasoning* (a stand-in for
``<think>...</think>``) and the ``[...]`` block is the *answer* (a stand-in for
``<answer>...</answer>``). The reward has two parts, exactly as in the paper:

* a **format reward** for using the reasoning/answer tags correctly, and
* an **accuracy reward** for the answer being right.

Nothing is supervised — the policy only ever sees these scalar rewards.
"""

from __future__ import annotations

import re

# Characters the policy can read/produce. Index 0 is reserved as padding.
VOCAB_CHARS = "0123456789+=" + "{}[]"
PAD = 0

# Tag legend for display (our single-char tags stand in for the paper's tags).
TAG_LEGEND = {"{": "<think>", "}": "</think>", "[": "<answer>", "]": "</answer>"}

# A well-formed response = a think block ``{...}`` then an answer block ``[...]``
# (the answer may be empty, which keeps the format easy to *discover*).
_FORMAT_RE = re.compile(r"^\{([^}]*)\}\[([^\]]*)\]")

R_FORMAT = 0.3        # using the tag structure at all
R_ANSWER_PRESENT = 0.2  # committing an actual number in the answer block
R_THINK = 0.2         # the think block showing the real computation "a+b"
R_ANSWER = 1.0        # the answer being exactly correct
R_MAX = R_FORMAT + R_ANSWER_PRESENT + R_THINK + R_ANSWER


class Vocab:
    def __init__(self):
        self.itos = ["\u0000"] + list(VOCAB_CHARS)
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids if i != PAD)


def prompt_str(a: int, b: int) -> str:
    return f"{a}+{b}="


def all_prompts(max_operand: int):
    return [(a, b) for a in range(max_operand + 1) for b in range(max_operand + 1)]


def reward(a: int, b: int, generated: str) -> dict:
    """Score a generated completion (format reward + accuracy reward).

    Rewards, as in the paper, are only:
      * ``R_FORMAT`` for using the ``{think}[answer]`` tags correctly,
      * ``R_THINK``  for the think block containing the actual computation,
      * ``R_ANSWER`` for the answer being exactly correct.
    Nothing is supervised — the policy only sees this scalar.
    """
    r = 0.0
    m = _FORMAT_RE.match(generated)
    format_ok = m is not None
    think_ok = False
    answer_ok = False
    if format_ok:
        r += R_FORMAT
        think, ans = m.group(1), m.group(2)
        if think == f"{a}+{b}":
            r += R_THINK
            think_ok = True
        if ans.isdigit() and len(ans) <= 2:
            r += R_ANSWER_PRESENT
            if int(ans) == a + b:
                r += R_ANSWER
                answer_ok = True
    return {"reward": r, "format_ok": format_ok, "think_ok": think_ok,
            "answer_ok": answer_ok}
