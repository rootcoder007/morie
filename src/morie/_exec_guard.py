# SPDX-License-Identifier: AGPL-3.0-or-later
"""Central execution guard for every dynamic-code sink in morie.

SECURITY / TRUST MODEL
----------------------
morie is a local analysis toolkit: code reaching these sinks is supplied
by the local user (CLI args, their own REPL input) or by an LLM the user
explicitly configured for the agent. Nothing in this package executes
remote or network-supplied code. Even so, every sink is funnelled through
this module so that:

* ``MORIE_NO_EXEC=1`` disables ALL dynamic execution (set it in CI,
  shared machines, or any automated/supply-chain context);
* LLM-generated code is AST-validated before it runs (import whitelist,
  no dunder access, no filesystem/process builtins);
* shell commands run WITHOUT a shell (``shlex`` token list) and only if
  the target binary is on an explicit allowlist.
"""

from __future__ import annotations

import ast
import builtins
import os
import shlex
import subprocess
import sys
from typing import Any


class ExecGuardError(RuntimeError):
    """Raised when the execution guard refuses to run code."""


def exec_disabled() -> bool:
    """True when dynamic execution is disabled via MORIE_NO_EXEC."""
    return os.environ.get("MORIE_NO_EXEC", "").strip() not in ("", "0")


def ensure_exec_allowed(feature: str = "dynamic code execution") -> None:
    """Raise ExecGuardError if MORIE_NO_EXEC is set."""
    if exec_disabled():
        raise ExecGuardError(
            f"{feature} is disabled because MORIE_NO_EXEC is set. "
            "Unset MORIE_NO_EXEC to allow it on this machine."
        )


# --------------------------------------------------------------------------
# Guarded exec() for LLM-generated setup/data code (agent tools).
# --------------------------------------------------------------------------

_ALLOWED_IMPORT_ROOTS = {
    "numpy", "pandas", "scipy", "math", "statistics", "random",
    "itertools", "collections", "datetime", "json", "re", "morie",
}

_BLOCKED_NAMES = {
    "eval", "exec", "compile", "__import__", "open", "input",
    "breakpoint", "globals", "locals", "vars", "getattr", "setattr",
    "delattr", "exit", "quit", "help", "memoryview", "object",
}

# Attribute names blocked even without a leading underscore: format-string
# escapes and known deserialization / native-load RCE gadgets on otherwise
# whitelisted libraries (pandas.read_pickle, numpy.ctypeslib.load_library,
# numpy.load(allow_pickle=True), joblib.load, ...).
_BLOCKED_ATTRS = {
    # format-string dunder-traversal escape
    "format", "format_map", "mro",
    # deserialization / native-load RCE gadgets on whitelisted libs
    "read_pickle", "to_pickle", "load_library", "ctypeslib",
    # process/shell gadgets (belt-and-braces; the modules aren't importable)
    "system", "popen", "fork", "check_output", "Popen",
}

_SAFE_BUILTIN_NAMES = (
    "abs", "all", "any", "bool", "bytes", "callable", "chr", "complex",
    "dict", "divmod", "enumerate", "filter", "float", "format",
    "frozenset", "hasattr", "hash", "hex", "int", "isinstance",
    "issubclass", "iter", "len", "list", "map", "max", "min", "next",
    "oct", "ord", "pow", "print", "range", "repr", "reversed", "round",
    "set", "slice", "sorted", "str", "sum", "tuple", "zip",
    "ArithmeticError", "AttributeError", "Exception", "IndexError",
    "KeyError", "TypeError", "ValueError", "ZeroDivisionError", "True",
    "False", "None",
)


def _guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
    root = name.split(".")[0]
    if root not in _ALLOWED_IMPORT_ROOTS:
        raise ExecGuardError(f"import of '{name}' is not allowed in guarded code")
    return __import__(name, *args, **kwargs)


def validate_source(code: str) -> ast.Module:
    """AST-validate untrusted (LLM-generated) Python source.

    Rejects: imports outside the whitelist, any dunder/underscore
    attribute access, and references to introspection / filesystem /
    process builtins.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ExecGuardError(f"invalid Python source: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [node.module or ""] if isinstance(node, ast.ImportFrom)
                else [a.name for a in node.names]
            )
            for name in names:
                root = name.split(".")[0]
                if root not in _ALLOWED_IMPORT_ROOTS:
                    raise ExecGuardError(f"import of '{name}' is not allowed")
        elif isinstance(node, ast.Attribute) and (
            node.attr.startswith("_") or node.attr in _BLOCKED_ATTRS
        ):
            raise ExecGuardError(f"access to attribute '{node.attr}' is not allowed")
        elif isinstance(node, ast.Name) and (
            node.id in _BLOCKED_NAMES or node.id.startswith("__")
        ):
            raise ExecGuardError(f"use of '{node.id}' is not allowed")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
            # closes the str.format dunder-traversal escape,
            # e.g. "{0.__class__.__mro__}".format(obj)
            raise ExecGuardError("string literals containing '__' are not allowed")
        elif (
            isinstance(node, ast.keyword)
            and node.arg == "allow_pickle"
            and not (isinstance(node.value, ast.Constant) and node.value.value is False)
        ):
            # numpy.load(allow_pickle=True) is a pickle-RCE vector
            raise ExecGuardError("allow_pickle=True is not allowed")
    return tree


def guarded_exec(code: str, namespace: dict[str, Any]) -> None:
    """Execute AST-validated code with restricted builtins.

    For code whose *author is not the local user* (i.e. LLM tool-call
    arguments). Raises ExecGuardError instead of running anything unsafe.
    """
    ensure_exec_allowed("guarded exec")
    tree = validate_source(code)
    safe_builtins = {
        name: getattr(builtins, name)
        for name in _SAFE_BUILTIN_NAMES
        if hasattr(builtins, name)
    }
    safe_builtins["__import__"] = _guarded_import
    namespace["__builtins__"] = safe_builtins
    exec(compile(tree, "<morie-guarded>", "exec"), namespace)  # noqa: S102


# --------------------------------------------------------------------------
# Safe eval for pure expressions (math formulas, boolean logic).
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Shell-free subprocess execution with a binary allowlist (agent tool).
# --------------------------------------------------------------------------

_SHELL_ALLOWLIST = {
    "ls", "cat", "head", "tail", "wc", "grep", "find", "pwd", "which",
    "echo", "date", "uname", "file", "stat", "du", "df", "sort", "uniq",
    "git", "python", "python3", "pip", "pip3", "R", "Rscript",
}


def safe_shell_run(
    command: str, *, timeout: int = 30, cwd: str | None = None
) -> subprocess.CompletedProcess[str]:
    """Run a command WITHOUT a shell, only if the binary is allowlisted.

    The command string is tokenized with shlex (no shell metacharacter
    interpretation, no pipes/redirection/substitution) and the first
    token must resolve to an allowlisted binary name.
    """
    ensure_exec_allowed("shell command execution")
    argv = shlex.split(command)
    if not argv:
        raise ExecGuardError("empty command")
    binary = os.path.basename(argv[0])
    if binary not in _SHELL_ALLOWLIST and argv[0] != sys.executable:
        allowed = ", ".join(sorted(_SHELL_ALLOWLIST))
        raise ExecGuardError(
            f"'{binary}' is not on the command allowlist ({allowed})"
        )
    return subprocess.run(  # noqa: S603 -- shlex-tokenized, allowlisted, shell=False
        argv, capture_output=True, text=True, timeout=timeout, cwd=cwd
    )
