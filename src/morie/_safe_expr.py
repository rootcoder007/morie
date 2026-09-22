# SPDX-License-Identifier: AGPL-3.0-or-later
"""AST-validated evaluation of one pure expression.

Shipped in the wheel, unlike :mod:`morie._exec_guard`: ``bexpr()`` and
``moncar()`` evaluate a user-written formula and were dead on every
``pip install`` because they imported the evaluator from the module the
wheel strips. Nothing here executes statements; only arithmetic, boolean
and comparison operators, literals, and attribute/call chains on the
names handed in are accepted, and no builtin is reachable.
"""

from __future__ import annotations

import ast
from typing import Any

_BLOCKED_NAMES = {
    "eval", "exec", "compile", "__import__", "open", "input",
    "breakpoint", "globals", "locals", "vars", "getattr", "setattr",
    "delattr", "exit", "quit", "help", "memoryview", "object",
}

_EXPR_NODES = (
    ast.Expression, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare,
    ast.Call, ast.Attribute, ast.Name, ast.Constant, ast.Tuple, ast.List,
    ast.Subscript, ast.IfExp, ast.Load,
    # operator tokens
    ast.And, ast.Or, ast.Not, ast.Invert, ast.UAdd, ast.USub,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Slice, ast.keyword,
)


def safe_eval_expr(expression: str, namespace: dict[str, Any] | None = None) -> Any:
    """Evaluate a single expression after strict AST validation.

    Only arithmetic/boolean/comparison operators, literals, names bound
    in ``namespace``, and attribute/call chains on those names (no
    underscore attributes) are allowed. No builtins are reachable.
    """
    namespace = dict(namespace or {})
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid expression: {exc}") from exc

    for node in ast.walk(tree):
        if not isinstance(node, _EXPR_NODES):
            raise ValueError(
                f"disallowed syntax in expression: {type(node).__name__}"
            )
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError(f"underscore attribute '{node.attr}' not allowed")
        if isinstance(node, ast.Name) and (
            node.id.startswith("__") or node.id in _BLOCKED_NAMES
        ):
            raise ValueError(f"name '{node.id}' not allowed")

    namespace["__builtins__"] = {}
    return eval(compile(tree, "<morie-expr>", "eval"), namespace)  # noqa: S307
