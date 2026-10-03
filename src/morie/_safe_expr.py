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
    "eval",
    "exec",
    "compile",
    "__import__",
    "open",
    "input",
    "breakpoint",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
    "exit",
    "quit",
    "help",
    "memoryview",
    "object",
}

_EXPR_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.Call,
    ast.Attribute,
    ast.Name,
    ast.Constant,
    ast.Tuple,
    ast.List,
    ast.Subscript,
    ast.IfExp,
    ast.Load,
    # operator tokens
    ast.And,
    ast.Or,
    ast.Not,
    ast.Invert,
    ast.UAdd,
    ast.USub,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Slice,
    ast.keyword,
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
            raise ValueError(f"disallowed syntax in expression: {type(node).__name__}")
        if isinstance(node, ast.Attribute) and (
            node.attr.startswith("_") or node.attr in ("format", "format_map", "mro")
        ):
            raise ValueError(f"attribute '{node.attr}' not allowed")
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
            # closes the str.format dunder-traversal walk,
            # e.g. "{0.__class__}".format(x), the same check _exec_guard has
            raise ValueError("string literals containing '__' are not allowed")
        if isinstance(node, ast.Name) and (node.id.startswith("__") or node.id in _BLOCKED_NAMES):
            raise ValueError(f"name '{node.id}' not allowed")

    return _Evaluator(namespace).visit(tree.body)


class _Evaluator:
    """Walk the validated tree and compute it: the same operators and node types
    :func:`safe_eval_expr` admits, with no call into the interpreter's eval()."""

    _BIN = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
        ast.Pow: lambda a, b: a**b,
    }
    _CMP = {
        ast.Eq: lambda a, b: a == b,
        ast.NotEq: lambda a, b: a != b,
        ast.Lt: lambda a, b: a < b,
        ast.LtE: lambda a, b: a <= b,
        ast.Gt: lambda a, b: a > b,
        ast.GtE: lambda a, b: a >= b,
    }

    def __init__(self, namespace: dict[str, Any]) -> None:
        self._ns = namespace

    def visit(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            try:
                return self._ns[node.id]
            except KeyError:
                raise NameError(f"name '{node.id}' is not defined") from None
        if isinstance(node, ast.Tuple):
            return tuple(self.visit(e) for e in node.elts)
        if isinstance(node, ast.List):
            return [self.visit(e) for e in node.elts]
        if isinstance(node, ast.UnaryOp):
            v = self.visit(node.operand)
            if isinstance(node.op, ast.Not):
                return not v
            if isinstance(node.op, ast.USub):
                return -v
            if isinstance(node.op, ast.UAdd):
                return +v
            return ~v
        if isinstance(node, ast.BinOp):
            return self._BIN[type(node.op)](self.visit(node.left), self.visit(node.right))
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                v = True
                for e in node.values:
                    v = self.visit(e)
                    if not v:
                        return v
                return v
            v = False
            for e in node.values:
                v = self.visit(e)
                if v:
                    return v
            return v
        if isinstance(node, ast.Compare):
            left = self.visit(node.left)
            for op, comp in zip(node.ops, node.comparators):
                right = self.visit(comp)
                if not self._CMP[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            return self.visit(node.body) if self.visit(node.test) else self.visit(node.orelse)
        if isinstance(node, ast.Attribute):
            return getattr(self.visit(node.value), node.attr)
        if isinstance(node, ast.Subscript):
            return self.visit(node.value)[self.visit(node.slice)]
        if isinstance(node, ast.Slice):
            return slice(
                None if node.lower is None else self.visit(node.lower),
                None if node.upper is None else self.visit(node.upper),
                None if node.step is None else self.visit(node.step),
            )
        if isinstance(node, ast.Call):
            fn = self.visit(node.func)
            args = [self.visit(a) for a in node.args]
            kwargs = {k.arg: self.visit(k.value) for k in node.keywords}
            return fn(*args, **kwargs)
        raise ValueError(f"node {type(node).__name__} is not allowed")
