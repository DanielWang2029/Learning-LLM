"""Transparent comparison agents for direct, CoT, and tool-integrated reasoning."""

from __future__ import annotations

import ast
import operator
from typing import Any


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}


def calculator(expression: str) -> int:
    """A deliberately tiny safe calculator, analogous to a Python-executor tool."""
    tree = ast.parse(expression, mode="eval")

    def visit(node: ast.AST) -> int:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](visit(node.left), visit(node.right))
        raise ValueError("only integer +, -, and * expressions are supported")

    return visit(tree.body)


def direct(problem: dict[str, Any]) -> dict[str, Any]:
    """Answer directly: a deliberately weak shortcut ignores the loose-pens adjustment."""
    answer = int(problem["a"]) * int(problem["b"])
    return {"answer": answer, "trace": f"Direct guess: {problem['a']} × {problem['b']} = {answer}"}


def chain_of_thought(problem: dict[str, Any]) -> dict[str, Any]:
    """Show an interpretable manual arithmetic failure on large multi-digit products.

    This is not a claim about Qwen2.5-Math's CoT performance.  It creates a
    controlled setting where a calculator's exact execution is observable.
    """
    a, b, c = int(problem["a"]), int(problem["b"]), int(problem["c"])
    product = a * b
    # A modeled carrying slip occurs only in the larger-number condition.
    reported_product = product if problem["scale"] == "small" else product + 10
    answer = reported_product + c
    return {
        "answer": answer,
        "trace": f"First multiply {a} × {b} = {reported_product}; then add {c} → {answer}",
    }


def tool_integrated_reasoning(problem: dict[str, Any]) -> dict[str, Any]:
    """Emit a tool call, execute it, then use the returned value in the answer."""
    expression = str(problem["expression"])
    result = calculator(expression)
    return {
        "answer": result,
        "tool_call": f"<tool>{expression}</tool>",
        "tool_result": result,
        "trace": f"I need exact arithmetic. {expression} evaluates to {result}; final answer: {result}.",
    }
