"""Synthetic structured-code corpus for the FIM reproduction.

Each function returns ``x <operator> <number>``.  The comment in the suffix
names the operation, so a FIM-trained decoder can use right context to recover
the missing expression.
"""

from __future__ import annotations

from itertools import product

OPS = (("+", "one"), ("+", "two"), ("-", "one"), ("*", "two"))


def examples() -> list[dict[str, list[str]]]:
    """Return all small code snippets and their prefix/middle/suffix splits."""
    rows = []
    for name, (op, word) in product(("add", "calc", "apply"), OPS):
        number = {"one": "1", "two": "2"}[word]
        rows.append(
            {
                "prefix": ["def", name, "(", "x", ")", ":", "return", "x"],
                "middle": [op, number],
                "suffix": ["#", word, op],
            }
        )
    return rows


def fim_tokens(row: dict[str, list[str]]) -> list[str]:
    """Paper Figure 3 order: prefix, suffix, then the target middle."""
    return ["<PRE>", *row["prefix"], "<SUF>", *row["suffix"], "<MID>", *row["middle"], "<END>"]


def next_tokens(row: dict[str, list[str]]) -> list[str]:
    """Ordinary left-to-right next-token training sequence."""
    return [*row["prefix"], *row["middle"], *row["suffix"], "<END>"]
