"""A battery of small, self-contained capability probes.

Sparks of AGI probes GPT-4 with a wide range of tasks (arithmetic, coding,
theory-of-mind, puzzles, tool use, ...) and reports where it succeeds and
fails. We reproduce that *methodology* at tiny scale with six transparent,
programmatic probes, each cast as a short token sequence a small GPT can be
trained on:

    [TASK_k] x1 x2 ... = y1 y2 ...

Only the answer region (after ``=``) is scored. The probes span a difficulty
range on purpose, so which ones a model passes depends on its size — exactly
the "capabilities appear with scale" story of the paper.
"""

from __future__ import annotations

import numpy as np

# --- Shared token vocabulary -------------------------------------------------
# 0..9 digits, then markers.
EQ = 10
PAD = 11
TASK_BASE = 12  # task tokens are TASK_BASE + task_index

TASKS = ["copy", "reverse", "sort", "add", "parity", "pattern"]
TASK_DESC = {
    "copy": "Echo the input (memory / identity)",
    "reverse": "Reverse the input (positional routing)",
    "pattern": "Continue an alternating pattern (induction)",
    "sort": "Sort the digits ascending (comparison)",
    "add": "Add two digits with carry (arithmetic)",
    "parity": "XOR of a bit string (sequential counting)",
}
VOCAB_SIZE = TASK_BASE + len(TASKS)

CONTENT_LEN = 6   # fixed slot for inputs (right-padded)
ANSWER_LEN = 4    # fixed slot for answers (right-padded)
SEQ_LEN = 1 + CONTENT_LEN + 1 + ANSWER_LEN  # task + content + '=' + answer = 12


def _one(task: str, rng: np.random.Generator):
    """Return (content_list, answer_list) for one example of ``task``."""
    if task == "copy":
        x = list(rng.integers(0, 10, size=4)); return x, list(x)
    if task == "reverse":
        x = list(rng.integers(0, 10, size=4)); return x, list(reversed(x))
    if task == "sort":
        x = list(rng.integers(0, 10, size=4)); return x, sorted(x)
    if task == "add":
        a, b = int(rng.integers(0, 10)), int(rng.integers(0, 10))
        s = a + b; return [a, b], [s // 10, s % 10]
    if task == "parity":
        x = list(rng.integers(0, 2, size=6)); return x, [int(sum(x) % 2)]
    if task == "pattern":
        p, q = int(rng.integers(0, 10)), int(rng.integers(0, 10))
        return [p, q, p, q, p, q], [p, q]
    raise ValueError(task)


def encode(task: str, content, answer):
    """Pack one example into fixed-length (seq, answer_mask)."""
    ti = TASKS.index(task)
    content = list(content) + [PAD] * (CONTENT_LEN - len(content))
    answer_padded = list(answer) + [PAD] * (ANSWER_LEN - len(answer))
    seq = [TASK_BASE + ti] + content + [EQ] + answer_padded
    mask = [0] * (1 + CONTENT_LEN + 1) + [1] * len(answer) + [0] * (ANSWER_LEN - len(answer))
    return np.array(seq, dtype=np.int64), np.array(mask, dtype=np.int64)


def make_batch(tasks, n_per_task: int, seed: int):
    """Build a mixed batch: seqs (N, SEQ_LEN) and answer masks (N, SEQ_LEN)."""
    rng = np.random.default_rng(seed)
    seqs, masks = [], []
    for task in tasks:
        for _ in range(n_per_task):
            c, a = _one(task, rng)
            s, m = encode(task, c, a)
            seqs.append(s); masks.append(m)
    return np.stack(seqs), np.stack(masks)


def make_task_eval(task: str, n: int, seed: int):
    rng = np.random.default_rng(seed)
    seqs, masks = [], []
    for _ in range(n):
        c, a = _one(task, rng)
        s, m = encode(task, c, a)
        seqs.append(s); masks.append(m)
    return np.stack(seqs), np.stack(masks)


def chance_accuracy(task: str) -> float:
    """Probability of guessing the whole answer right by chance."""
    if task == "parity":
        return 0.5
    if task == "add":
        # tens digit in {0,1}, ones digit in 0..9 but coupled; approx via samples
        return 1 / 19  # 19 distinct sums 0..18 -> exact-match by luck
    if task == "pattern":
        return (1 / 10) ** 2
    return (1 / 10) ** 4  # copy/reverse/sort over 4 digits
