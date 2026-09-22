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
import types
from typing import Any

from morie._safe_expr import (  # noqa: F401  (shipped evaluator)
    _BLOCKED_NAMES,
    _EXPR_NODES,
    safe_eval_expr,
)


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
# Per-risk-class opt-in knobs.
#
# Each high-impact capability defaults to the SAFE behaviour. The risky
# path runs only when its env var is set truthfully ("1"/"true"/"yes"/"on").
# The knob NAME is the contract shared with bin/morie (a shell script that
# reads the same variables directly). Keep the names in sync there.
# --------------------------------------------------------------------------

_TRUE = {"1", "true", "yes", "on"}


def _enabled(var: str) -> bool:
    return os.environ.get(var, "").strip().lower() in _TRUE


def remote_install_allowed() -> bool:
    """True to allow remote-script installers (Ollama ``curl | sh``)."""
    return _enabled("MORIE_ALLOW_REMOTE_INSTALL")


def checkpoint_trusted() -> bool:
    """True to allow ``torch.load(weights_only=False)`` on a checkpoint."""
    return _enabled("MORIE_TRUST_CHECKPOINT")


def rc_autoload_allowed() -> bool:
    """True to auto-source the ESML_RC shell config on every invocation."""
    return _enabled("MORIE_ALLOW_RC")


def cron_allowed() -> bool:
    """True to allow writing to the user crontab (persistence)."""
    return _enabled("MORIE_ALLOW_CRON")


# knob name -> (getter, one-line description of what enabling it permits)
_KNOBS = {
    "MORIE_NO_EXEC": (
        exec_disabled,
        "when set: ALL dynamic execution (REPL/exec/shell) is disabled",
    ),
    "MORIE_ALLOW_REMOTE_INSTALL": (
        remote_install_allowed,
        "when set: `curl | sh` remote installers may run",
    ),
    "MORIE_TRUST_CHECKPOINT": (
        checkpoint_trusted,
        "when set: torch.load may execute pickled code in checkpoints",
    ),
    "MORIE_ALLOW_RC": (
        rc_autoload_allowed,
        "when set: the ESML_RC shell config is auto-sourced",
    ),
    "MORIE_ALLOW_CRON": (
        cron_allowed,
        "when set: `morie cron` may write to your crontab",
    ),
}


def knob_status() -> list[dict[str, Any]]:
    """Structured report of every trust knob and its current state.

    Used by the ``doctor`` module so the active trust posture is visible
    on startup.
    """
    return [
        {"name": name, "enabled": bool(getter()), "detail": detail}
        for name, (getter, detail) in _KNOBS.items()
    ]


# --------------------------------------------------------------------------
# Guarded exec() for LLM-generated setup/data code (agent tools).
# --------------------------------------------------------------------------

# "morie" is deliberately NOT here. morie's own modules import os,
# subprocess, ctypes, importlib and pickle as ordinary attributes, so
# `import morie.container as m; m.subprocess.run([...])` was a two-line
# escape and no attribute blocklist could close it (importlib.import_module
# defeats any list by construction). Guarded code gets morie's arrays,
# frames and dataset loaders through guarded_namespace() instead.
_ALLOWED_IMPORT_ROOTS = {
    "numpy", "pandas", "scipy", "math", "statistics", "random",
    "itertools", "collections", "datetime", "json", "re",
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
    # stdlib modules that library code re-exports as attributes; the
    # module proxy below refuses them by type, this refuses them by name
    "os", "sys", "subprocess", "shutil", "importlib", "ctypes", "pickle",
    "socket", "builtins", "import_module", "CDLL", "sysconfig",
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


class _GuardedModule:
    """Read-only view of a morie module for guarded code.

    Attribute access refuses underscore names, the attribute blocklist,
    and any value that is a module outside morie's own namespace. A
    morie submodule comes back wrapped the same way, so no chain of
    attributes reaches os, subprocess, ctypes or importlib.
    """

    __slots__ = ("_mod",)

    def __init__(self, mod: types.ModuleType) -> None:
        object.__setattr__(self, "_mod", mod)

    def __getattribute__(self, name: str) -> Any:
        if name.startswith("_") or name in _BLOCKED_ATTRS:
            raise ExecGuardError(f"access to attribute '{name}' is not allowed")
        val = getattr(object.__getattribute__(self, "_mod"), name)
        if isinstance(val, types.ModuleType):
            if (val.__name__ + ".").startswith("morie."):
                return _GuardedModule(val)
            raise ExecGuardError(
                f"access to module '{val.__name__}' is not allowed in guarded code"
            )
        return val

    def __setattr__(self, name: str, value: Any) -> None:
        raise ExecGuardError("guarded modules are read-only")

    def __repr__(self) -> str:
        return f"<guarded {object.__getattribute__(self, '_mod').__name__}>"


def guarded_namespace() -> dict[str, Any]:
    """The names guarded code may use.

    ``np`` and ``pd`` are morie's own array and frame modules behind
    :class:`_GuardedModule`; ``load_dataset`` and ``DATASET_CATALOG``
    are the bundled-data entry points. This replaces ``import morie``
    inside guarded code, which is refused.
    """
    from morie.data import DATASET_CATALOG, load_dataset
    from morie.fn import _array_core, _frame_core

    return {
        "np": _GuardedModule(_array_core),
        "pd": _GuardedModule(_frame_core),
        "load_dataset": load_dataset,
        "DATASET_CATALOG": DATASET_CATALOG,
    }


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
