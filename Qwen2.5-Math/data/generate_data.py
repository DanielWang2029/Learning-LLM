"""Small deterministic arithmetic word problems for the TIR demonstration."""

from __future__ import annotations


def problems() -> list[dict[str, int | str]]:
    """Problems of two scales; all have the same exact arithmetic semantics."""
    rows = []
    for scale, triples in {
        "small": [(4, 3, 2), (7, 5, 4), (9, 2, 6), (8, 6, 3)],
        "large": [(137, 46, 29), (284, 73, 58), (419, 38, 67), (156, 91, 44)],
    }.items():
        for a, b, c in triples:
            # Purchasing b boxes of a items, then adding c loose items.
            rows.append(
                {
                    "scale": scale,
                    "a": a, "b": b, "c": c,
                    "expression": f"({a} * {b}) + {c}",
                    "answer": a * b + c,
                    "question": f"A shop packs {b} boxes of {a} pens and adds {c} loose pens. How many pens?",
                }
            )
    return rows
