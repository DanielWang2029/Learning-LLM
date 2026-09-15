"""A toy arithmetic reasoning task with a correctness reward (DeepSeekMath §4).

DeepSeekMath trains a model to solve math problems and rewards it for reaching
the correct answer. We reproduce that setting at tiny scale: the prompt is an
addition problem ``a + b =`` and the policy must GENERATE the answer digits.

The reward is outcome-based, like the paper's rule-based math reward, but lightly
graded so that a *group* of sampled answers has some reward variance to learn
from (pure 0/1 rewards give a zero advantage whenever a whole group misses, the
classic RL cold-start problem):

    1.0                              if the answer equals a + b exactly
    0.2 + 0.8 * (fraction correct)   if it has the right number of digits
    0.05                             if it is digits but the wrong length
    0.0                              otherwise (non-digit / malformed)

An exact answer therefore scores the full 1.0. Everything is expressed over a
tiny fixed vocabulary so a small policy can learn it from reward alone under
GRPO — no supervised answer labels are ever used.
"""

from __future__ import annotations

from typing import List, Tuple

# Vocabulary: digits 0-9 then special/format tokens.
DIGITS = [str(d) for d in range(10)]
PLUS, EQ, BOS, EOS, PAD = 10, 11, 12, 13, 14
VOCAB_SIZE = 15
ID_TO_STR = {**{i: str(i) for i in range(10)},
             PLUS: "+", EQ: "=", BOS: "<s>", EOS: "</s>", PAD: "_"}


def all_prompts(max_operand: int) -> List[Tuple[int, int]]:
    """Every (a, b) with 0 <= a, b <= max_operand."""
    return [(a, b) for a in range(max_operand + 1) for b in range(max_operand + 1)]


def encode_prompt(a: int, b: int) -> List[int]:
    """`a + b =` as token ids, e.g. 12 + 3 -> [BOS,1,2,PLUS,3,EQ]."""
    toks = [BOS] + [int(c) for c in str(a)] + [PLUS] + [int(c) for c in str(b)] + [EQ]
    return toks


def decode_answer(token_ids: List[int]) -> str:
    """Turn generated answer tokens (up to EOS) into a digit string."""
    out = []
    for t in token_ids:
        if t == EOS:
            break
        if 0 <= t <= 9:
            out.append(str(t))
        else:
            # a non-digit token in the answer region is a formatting error
            out.append("?")
    return "".join(out)


def reward(a: int, b: int, answer_str: str) -> float:
    """Outcome-based, lightly-graded reward for the generated answer string."""
    target = str(a + b)
    if answer_str == target:
        return 1.0
    if not answer_str.isdigit():
        return 0.0
    if len(answer_str) == len(target):
        correct = sum(x == y for x, y in zip(answer_str, target))
        return 0.2 + 0.8 * (correct / len(target))
    return 0.05


def max_answer_len(max_operand: int) -> int:
    return len(str(2 * max_operand))
