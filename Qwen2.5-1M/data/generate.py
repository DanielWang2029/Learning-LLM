"""Synthetic needle-in-a-haystack sequences; only NumPy is used to make them."""
import numpy as np


def make_needles(length: int, count: int, seed: int) -> list[tuple[np.ndarray, int]]:
    rng = np.random.default_rng(seed + length)
    examples = []
    for _ in range(count):
        # 1 is the unique key-like needle; zero represents unrelated hay.
        tokens = np.zeros(length, dtype=np.int64)
        needle = int(rng.integers(0, length - 1))
        tokens[needle] = 1
        examples.append((tokens, needle))
    return examples
