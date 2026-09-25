"""Tests for vlfctn.value_function_eval (value of a treatment regime)."""

import math
import statistics

import pytest

from morie.fn.vlfctn import value_function_eval


def _data(n=80):
    x = [math.sin(1.3 * k) for k in range(n)]
    e = [1 / (1 + math.exp(-0.5 * v)) for v in x]
    d = [1.0 if ((29 * k + 3) % 61 + 0.5) / 61 < p else 0.0 for k, p in enumerate(e)]
    y = [dd * v + 0.3 * math.cos(3.1 * k) for k, (dd, v) in enumerate(zip(d, x))]
    return y, d, x, e


def _ols(xs, ys):
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    b = sum((a - mx) * (c - my) for a, c in zip(xs, ys)) / sum((a - mx) ** 2 for a in xs)
    return my - b * mx, b


@pytest.mark.parametrize("method", ["aipw", "ipw", "regression"])
def test_vlfctn_basic(method):
    """With the propensity supplied: mu_a by OLS within arm a; psi_i =
    mu_d(X_i) + 1{A_i = d(X_i)}/pi_i (Y_i - mu_d(X_i)) (AIPW), its IPW or
    regression part; value = mean psi, se = sd(psi)/sqrt n."""
    y, d, x, e = _data()
    g = [1.0 if v > 0 else 0.0 for v in x]
    a1 = _ols([v for v, dd in zip(x, d) if dd], [yy for yy, dd in zip(y, d) if dd])
    a0 = _ols([v for v, dd in zip(x, d) if not dd], [yy for yy, dd in zip(y, d) if not dd])

    def psi(rule):
        out = []
        for yi, di, xi, ei, ri in zip(y, d, x, e, rule):
            mu = (a1 if ri else a0)[0] + (a1 if ri else a0)[1] * xi
            pi = ei if di else 1 - ei
            f = 1.0 if di == ri else 0.0
            out.append({"ipw": f / pi * yi, "regression": mu, "aipw": mu + f / pi * (yi - mu)}[method])
        return out

    ps = psi(g)
    r = value_function_eval(y, d, [[v] for v in x], g, propensity=e, method=method)
    # 1e-10: the arm regressions are solved by different least-squares
    # routines here and in the module; they agree to ~1e-12
    assert r["value"] == pytest.approx(statistics.fmean(ps), abs=1e-10)
    assert r["se"] == pytest.approx(statistics.stdev(ps) / math.sqrt(len(y)), rel=1e-10)
    assert r["value_treat_all"] == pytest.approx(statistics.fmean(psi([1.0] * len(y))), abs=1e-10)
    assert r["value_treat_none"] == pytest.approx(statistics.fmean(psi([0.0] * len(y))), abs=1e-10)
    assert r["n_following"] == sum(1 for a, b in zip(d, g) if a == b)


def test_vlfctn_edge():
    """Treating where x > 0 beats both static rules here; non-binary
    treatment, a non-binary regime and an unknown method raise."""
    y, d, x, e = _data()
    g = [1.0 if v > 0 else 0.0 for v in x]
    r = value_function_eval(y, d, [[v] for v in x], g, propensity=e)
    assert r["beats_static"] is True
    with pytest.raises(ValueError):
        value_function_eval(y, [0.5] * len(y), [[v] for v in x], g)
    with pytest.raises(ValueError):
        value_function_eval(y, d, [[v] for v in x], [0.5] * len(y))
    with pytest.raises(ValueError):
        value_function_eval(y, d, [[v] for v in x], g, method="dr")


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.vlfctn as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
