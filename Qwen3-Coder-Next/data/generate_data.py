"""Verifiable toy coding specifications and their hidden unit tests."""

from __future__ import annotations


def task_specs() -> list[dict[str, object]]:
    """Each spec names a target program; tests are the only reward oracle."""
    return [
        {"name": "increment", "program": "return x + 1", "tests": [2, 5, 9]},
        {"name": "add_three", "program": "return x + 3", "tests": [2, 5, 9]},
        {"name": "double", "program": "return x * 2", "tests": [2, 5, 9]},
        {"name": "triple", "program": "return x * 3", "tests": [2, 5, 9]},
        {"name": "square", "program": "return x * x", "tests": [2, 5, 9]},
        {"name": "negate", "program": "return -x", "tests": [2, 5, 9]},
    ]
