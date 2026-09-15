"""A tiny environment with two REAL tools: a knowledge base and a calculator.

ReAct (Yao et al., 2022) interleaves reasoning with *acting* in an environment.
Here the environment exposes two tools the agent can genuinely call:

* ``Lookup[entity, attribute]`` — query a small key-value knowledge base.
* ``Calc[expression]``          — evaluate an arithmetic expression.

Answering the demo questions requires *multi-hop* tool use: e.g. look up which
country a person lives in, then look up that country's capital — the second
action depends on the first observation.
"""

from __future__ import annotations

import ast
import operator as op

# ---------------------------------------------------------------------------
# Knowledge base: entities -> {attribute: value}. Some values reference other
# entities (e.g. a person's "country"), which is what makes questions multi-hop.
# ---------------------------------------------------------------------------
KB: dict[str, dict[str, object]] = {
    # people
    "Alice": {"country": "France", "age": 30, "pet": "dog"},
    "Bob":   {"country": "Japan",  "age": 25, "pet": "cat"},
    "Carol": {"country": "Brazil", "age": 41, "pet": "dog"},
    "Dave":  {"country": "Canada", "age": 52, "pet": "cat"},
    # countries (population in millions)
    "France": {"capital": "Paris",    "population": 68},
    "Japan":  {"capital": "Tokyo",    "population": 125},
    "Brazil": {"capital": "Brasilia", "population": 214},
    "Canada": {"capital": "Ottawa",   "population": 39},
    # things
    "bicycle":  {"wheels": 2},
    "car":      {"wheels": 4},
    "tricycle": {"wheels": 3},
    # animals
    "dog": {"legs": 4},
    "cat": {"legs": 4},
}


class ToolResult:
    """A tool observation, including whether the call succeeded."""

    def __init__(self, ok: bool, value: object, text: str) -> None:
        self.ok = ok
        self.value = value
        self.text = text

    def __repr__(self) -> str:
        return self.text


def lookup(entity: str, attribute: str) -> ToolResult:
    """Knowledge-base lookup tool."""
    ent = KB.get(entity)
    if ent is None:
        return ToolResult(False, None, f"no entity '{entity}' in the knowledge base")
    if attribute not in ent:
        return ToolResult(False, None, f"'{entity}' has no attribute '{attribute}'")
    val = ent[attribute]
    return ToolResult(True, val, f"{entity}.{attribute} = {val}")


# Safe arithmetic evaluation (no eval of arbitrary Python).
_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
    ast.Div: op.truediv, ast.USub: op.neg, ast.Pow: op.pow,
}


def _eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


def calc(expression: str) -> ToolResult:
    """Calculator tool: evaluate a numeric expression like '3*2 + 2*4'."""
    try:
        val = _eval(ast.parse(expression, mode="eval").body)
        if isinstance(val, float) and val.is_integer():
            val = int(val)
        return ToolResult(True, val, f"{expression} = {val}")
    except Exception as e:  # noqa: BLE001
        return ToolResult(False, None, f"calc error: {e}")
