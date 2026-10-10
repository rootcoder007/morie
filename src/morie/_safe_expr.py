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

# attribute names that lead out of arithmetic: modules the array core re-exports, the
# interpreter's own objects, and anything that runs code or touches the file system
_BLOCKED_ATTRS = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "open",
        "input",
        "breakpoint",
        "exit",
        "quit",
        "system",
        "popen",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "dir",
        "import_module",
        "bltns",
        "builtins",
        "sys",
        "os",
        "re",
        "enum",
        "importlib",
        "subprocess",
        "types",
        "ctypes",
        "io",
        "pathlib",
        "shutil",
        "socket",
        "codecs",
        "pickle",
        "marshal",
        "gc",
        "inspect",
    }
)
# file input and output: a formula computes, it never reads or writes a file (np.savez wrote an
# arbitrary path from a moncar formula; np.load read one back; frames and arrays have writers too)
_IO_PREFIXES = ("save", "load", "dump", "read", "write", "to_")
_IO_ALLOWED = frozenset({"to_numpy", "to_list", "to_dict"})
_IO_NAMES = frozenset(
    {
        "tofile",
        "memmap",
        "open_memmap",
        "fromfile",
        "fromregex",
        "genfromtxt",
        "unlink",
        "remove",
        "rmdir",
        "rmtree",
        "mkdir",
        "makedirs",
        "rename",
        "chmod",
        "chown",
        "touch",
        "symlink_to",
        "hardlink_to",
        "link_to",
    }
)


def _is_io_attr(name: str) -> bool:
    return name in _IO_NAMES or (name.startswith(_IO_PREFIXES) and name not in _IO_ALLOWED)


# builtins an expression may call once it has reached them through a value's attribute
_SAFE_BUILTINS = frozenset({"abs", "min", "max", "round", "sum", "len", "float", "int", "bool", "pow", "divmod"})

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
            node.attr.startswith("_")
            or node.attr in ("format", "format_map", "mro")
            or node.attr in _BLOCKED_ATTRS
            or _is_io_attr(node.attr)
        ):
            raise ValueError(f"attribute '{node.attr}' not allowed")
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
            # closes the str.format dunder-traversal walk,
            # e.g. "{0.__class__}".format(x), the same check _exec_guard has
            raise ValueError("string literals containing '__' are not allowed")
        if isinstance(node, ast.Name) and (node.id.startswith("__") or node.id in _BLOCKED_NAMES):
            raise ValueError(f"name '{node.id}' not allowed")

    return _Evaluator(namespace).visit(tree.body)


# a formula is arithmetic on data, not a way to stall the machine: 10**10**10 ran unbounded and
# "a" * 10**8 allocated 100 MB; integer powers and sequence repetition get a ceiling
_MAX_INT_BITS = 10_000
_MAX_REPEAT = 1_000_000


def _capped_pow(a, b):
    if (
        isinstance(a, int)
        and isinstance(b, int)
        and not isinstance(a, bool)
        and b > 0
        and abs(a) > 1
        and b * abs(a).bit_length() > _MAX_INT_BITS
    ):
        raise ValueError(f"result too large: an integer power above {_MAX_INT_BITS} bits")
    return a**b


def _capped_mult(a, b):
    for seq, n in ((a, b), (b, a)):
        if isinstance(seq, str | bytes | list | tuple) and isinstance(n, int) and len(seq) * n > _MAX_REPEAT:
            raise ValueError(f"result too large: more than {_MAX_REPEAT:,} repeated items")
    return a * b


class _Evaluator:
    """Walk the validated tree and compute it: the same operators and node types
    :func:`safe_eval_expr` admits, without handing the text to the interpreter."""

    _BIN = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: _capped_mult(a, b),
        ast.Div: lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
        ast.Pow: lambda a, b: _capped_pow(a, b),
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

    def _attribute(self, base: Any, attr: str) -> Any:
        """``base.attr`` only while the result stays arithmetic: never a module, a class or an
        interpreter builtin (``np.re.enum.bltns.eval`` walked out through the array core's
        re-exports; the static check cannot see what a name resolves to)."""
        import types

        if isinstance(base, types.ModuleType) and base not in self._ns.values():
            raise ValueError(f"attribute access on module '{base.__name__}' is not allowed")
        if _is_io_attr(attr):
            raise ValueError(f"'{attr}' reads or writes files and is not allowed in an expression")
        value = getattr(base, attr)
        if isinstance(value, types.ModuleType):
            raise ValueError(f"'{attr}' is a module and may not be used in an expression")
        if (
            isinstance(value, type)
            and getattr(value, "__module__", "") == "builtins"
            and value.__name__ not in _SAFE_BUILTINS
        ):
            raise ValueError(f"'{attr}' is not allowed in an expression")
        if isinstance(value, types.BuiltinFunctionType) and value.__name__ not in _SAFE_BUILTINS:
            owner = getattr(value, "__self__", None)
            # a method of a value ("ab".upper) and a function of a module the caller put in the
            # namespace (math.sin) are arithmetic; a function of the builtins module, of a module
            # reached through an attribute, or of a class (str.format) is the interpreter
            if owner is None or isinstance(owner, type):
                raise ValueError(f"'{attr}' is not allowed in an expression")
            if isinstance(owner, types.ModuleType) and (
                owner.__name__ == "builtins" or not any(owner is v for v in self._ns.values())
            ):
                raise ValueError(f"'{attr}' is not allowed in an expression")
        return value

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
            return self._attribute(self.visit(node.value), node.attr)
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
