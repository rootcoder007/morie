# SPDX-License-Identifier: AGPL-3.0-or-later
"""morie._safe_expr ships in the wheel; bexpr() and moncar() depend on it."""
import pytest

from morie._safe_expr import safe_eval_expr


def test_arithmetic_and_names():
    assert safe_eval_expr("2 ** 3 + y", {"y": 1}) == 9


@pytest.mark.parametrize("expr", ["__import__('os')", "open('x')", "x.__class__", "(lambda: 1)()"])
def test_refuses_escapes(expr):
    with pytest.raises(ValueError):
        safe_eval_expr(expr, {"x": 1})


def test_bexpr_and_moncar_do_not_need_exec_guard(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "morie._exec_guard", None)  # import would fail
    from morie.fn import boolean_eval
    from morie.fn.moncar import moncar

    assert boolean_eval("True and False", {}).value == 0
    r = moncar()
    assert abs(float(r.value) - 1 / 3) < 0.02
