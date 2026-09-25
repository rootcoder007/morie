"""Tests for morie.fn.pptst — Phillips-Perron unit root test."""

from morie.fn import _array_core as np
import pytest

from morie.fn.pptst import pp_test, pptst


def _ar1(phi: float, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.standard_normal()
    return x


def test_returns_test_result():
    """Return type has the standard TestResult interface."""
    x = _ar1(0.5, 100, seed=10)
    r = pp_test(x)
    assert hasattr(r, "statistic")
    assert hasattr(r, "p_value")
    assert 0.0 <= r.p_value <= 1.0


def test_random_walk_high_pvalue():
    """Unit root series — PP should NOT reject H0 (p >= 0.05)."""
    x = _ar1(1.0, 200, seed=1)
    r = pp_test(x)
    assert r.p_value >= 0.05, f"Expected p >= 0.05 for RW, got {r.p_value}"


def test_stationary_ar1_rejected():
    """Stationary AR(1) — PP should reject H0 (p <= 0.10)."""
    x = _ar1(0.2, 200, seed=2)
    r = pp_test(x)
    assert r.p_value <= 0.10, f"Expected p <= 0.10, got {r.p_value}"


def test_explicit_lags():
    """Explicit lag count is stored in extra."""
    x = _ar1(0.5, 100, seed=3)
    r = pp_test(x, lags=5)
    assert r.extra["lags"] == 5


def test_critical_values_present():
    x = _ar1(0.5, 80, seed=4)
    r = pp_test(x)
    cv = r.extra["critical_values"]
    assert "1%" in cv and "5%" in cv and "10%" in cv
    assert cv["1%"] < cv["5%"] < cv["10%"]


def test_lrvar_positive():
    """Long-run variance estimate should be positive."""
    x = _ar1(0.4, 100, seed=5)
    r = pp_test(x)
    assert r.extra["lrvar"] > 0.0


def test_too_short_raises():
    with pytest.raises(ValueError):
        pp_test([1.0, 2.0, 3.0])


def test_alias():
    assert pptst is pp_test


def test_z_tau_matches_the_urca_formula():
    """urca::ur.pp(type = "Z-tau", model = "constant"), recomputed:
    regress y_t on (1, y_{t-1}); s = SSR/n; Bartlett long-run variance
    with L lags; Z = sqrt(s/sig) t - (sig - s)/(2 sig) sqrt(sig/ybar2)."""
    import math

    x = [float(v) for v in _ar1(0.6, 90, seed=4)]
    n, L = len(x) - 1, 3
    yl, yc = x[:-1], x[1:]
    ml, mc = sum(yl) / n, sum(yc) / n
    sxx = sum((v - ml) ** 2 for v in yl)
    rho = sum((a - ml) * (c - mc) for a, c in zip(yl, yc)) / sxx
    u = [c - (mc - rho * ml) - rho * a for a, c in zip(yl, yc)]
    t = (rho - 1) / math.sqrt(sum(e * e for e in u) / (n - 2) / sxx)
    s = sum(e * e for e in u) / n
    sig = s + 2 / n * sum((1 - l / (L + 1)) * sum(u[i] * u[i - l] for i in range(l, n))
                          for l in range(1, L + 1))
    yb2 = sum((c - mc) ** 2 for c in yc) / n ** 2
    z = math.sqrt(s / sig) * t - 0.5 * (sig - s) / sig * math.sqrt(sig / yb2)
    r = pp_test(x, lags=L)
    assert r.statistic == pytest.approx(z, rel=1e-12)
    assert r.extra["critical_values"]["5%"] == pytest.approx(-2.8621 - 2.738 / n - 8.36 / n ** 2, rel=1e-15)


# --- appended: the module worked example as a gate -----------

import doctest as _doctest
import importlib as _importlib

_doctest_module = _importlib.import_module("morie.fn.pptst")


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
