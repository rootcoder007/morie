# SPDX-License-Identifier: AGPL-3.0-or-later
"""What the published wheel must not contain: dynamic code execution, shell=True, a network call on import."""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "morie"
# the interactive layer is excluded from the wheel (pyproject wheel.exclude) and is the one place that runs user code
LAYER = {"polyglot.py", "agent.py", "tui.py", "_exec_guard.py", "repl_init.py"}


def _shipped_py():
    for p in SRC.rglob("*.py"):
        rel = p.relative_to(SRC)
        if (
            rel.name in LAYER
            or rel.parts[0] == "fn"
            and rel.name not in ("__init__.py",)
            and not rel.name.startswith("_")
        ):
            continue
        yield p


def test_no_eval_or_exec_calls_in_the_shipped_package():
    hits = []
    for p in _shipped_py():
        tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in ("eval", "exec", "compile")
            ):
                hits.append(f"{p.relative_to(SRC)}:{node.lineno}:{node.func.id}")
    assert hits == [], hits


def test_no_shell_true_in_the_shipped_package():
    hits = []
    for p in _shipped_py():
        tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                        hits.append(f"{p.relative_to(SRC)}:{node.lineno}")
    assert hits == [], hits


def test_import_does_not_start_the_update_check(monkeypatch):
    import importlib

    calls = []
    import morie._update_check as uc

    monkeypatch.setattr(uc, "maybe_notify", lambda v: calls.append(v))
    # a fresh import of morie, then the original module objects back: tests that run later on
    # this worker hold references to them (an identity check in test_exec_guard failed otherwise)
    saved = {n: m for n, m in sys.modules.items() if n == "morie" or n.startswith("morie.")}
    try:
        for name in saved:
            if name not in ("morie._update_check",):
                sys.modules.pop(name, None)
        importlib.import_module("morie")
        assert calls == []
    finally:
        for name in [n for n in list(sys.modules) if n == "morie" or n.startswith("morie.")]:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


def test_safe_expr_evaluates_without_eval():
    from morie._safe_expr import safe_eval_expr

    assert safe_eval_expr("2 ** 3 + y", {"y": 1}) == 9
    assert safe_eval_expr("a if x > 1 else b", {"x": 2, "a": "yes", "b": "no"}) == "yes"
    assert safe_eval_expr("xs[1:3]", {"xs": [1, 2, 3, 4]}) == [2, 3]
    assert safe_eval_expr("(1, 2) == (1, 2) and not False", {}) is True
    assert safe_eval_expr("s.upper()", {"s": "ab"}) == "AB"
    assert safe_eval_expr("1 < 2 < 3", {}) is True and safe_eval_expr("1 < 3 < 2", {}) is False
    with pytest.raises(ValueError):
        safe_eval_expr("__import__('os')", {})
    with pytest.raises(NameError):
        safe_eval_expr("zz + 1", {})


def test_fn_sources_load_through_zipimport_even_without_a_cache_dir(monkeypatch, tmp_path):
    import morie.fn as fn

    xz = Path(fn.__file__).with_name("_fnsrc.json.xz")
    zipf = Path(fn.__file__).with_name("_fnsrc.zip")
    if not xz.is_file() or zipf.is_file():
        pytest.skip("layout without the xz archive (source checkout)")
    monkeypatch.setattr(fn, "_candidate_cache_dirs", lambda: [str(tmp_path / "ro" / "nope")])
    (tmp_path / "ro").mkdir()
    (tmp_path / "ro").chmod(0o500)
    before = list(fn.__path__)
    meta_before = list(sys.meta_path)
    try:
        fn._install_fnsrc()
    finally:
        (tmp_path / "ro").chmod(0o700)
    try:
        # no writable cache dir: nothing is written anywhere (no process-private zip that
        # nothing cleans up); the sources are served from memory by a meta-path finder
        assert [p for p in fn.__path__ if p not in before] == []
        assert list((tmp_path / "ro").iterdir()) == []
        # the finder is installed once per process; an earlier import may have added it
        finder = next((f for f in sys.meta_path if isinstance(f, fn._Finder)), None)
        assert finder is not None
        short = next(iter(fn._decompress_fnsrc(str(xz))))
        spec = finder.find_spec(f"morie.fn.{short}")
        assert spec is not None and spec.origin == f"morie-fnsrc:{short}.py"
        src = spec.loader.get_data(spec.origin).decode("utf-8")
        assert src.strip()
        # a SourceLoader: the import system compiles the module, no eval/exec in morie
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod.__name__ == f"morie.fn.{short}"
        assert finder.find_spec("morie.fn.__no_such_function__") is None
        assert finder.find_spec("os.path") is None
    finally:
        sys.meta_path[:] = meta_before


def test_morie_exec_runs_in_a_child_interpreter(tmp_path):
    import os
    import subprocess

    env = {**os.environ, "MORIE_NO_UPDATE_CHECK": "1", "PYTHONPATH": str(SRC.parent)}
    r = subprocess.run(
        [
            sys.executable,
            "-c",
            "from morie.runner import main; raise SystemExit(main())",
            "exec",
            "print(6 * 7); import sys; sys.exit(3)",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    if "not bundled" in r.stdout:
        pytest.skip("interactive layer absent")
    assert "42" in r.stdout and r.returncode == 3
