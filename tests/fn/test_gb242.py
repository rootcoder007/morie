"""Tests for gb242.gibbons_order_pdf."""

import math

from morie.fn import _array_core as np

from morie.fn.gb242 import gibbons_order_pdf


def test_gb242_basic():
    """Test basic functionality against the documented formula."""
    rng = np.random.default_rng(42)

    # Parent distribution: standard normal.
    x = float(rng.normal(0.0, 1.0))

    # Order-statistic index r and sample size n, 1 <= r <= n.
    r = 10
    n = 100

    # Provide F_X(x) and f_X(x) as the values they take at x (floats),
    # computed independently from the parent distribution.
    from math import erf, sqrt, pi, exp

    fx_val = float(0.5 * (1.0 + erf(x / sqrt(2.0))))
    dx_val = float(exp(-0.5 * x * x) / sqrt(2.0 * pi))

    result = gibbons_order_pdf(x, r, n, fx_val, dx_val)

    # Result is a mapping with the keys documented in the docstring.
    assert isinstance(result, dict) or hasattr(result, "__getitem__")
    for key in ("pdf", "coef", "fx", "dx", "r", "n", "method"):
        assert key in result, f"missing key {key!r}"

    # Re-derive the expected density from the literature formula
    # using only plain arithmetic on the same inputs.
    expected_coef = float(
        math.factorial(n) / (math.factorial(r - 1) * math.factorial(n - r))
    )
    expected_pdf = (
        expected_coef
        * fx_val ** (r - 1)
        * (1.0 - fx_val) ** (n - r)
        * dx_val
    )

    assert result["r"] == r
    assert result["n"] == n
    assert math.isclose(result["coef"], expected_coef, rel_tol=1e-12, abs_tol=0.0)
    assert math.isclose(result["fx"], fx_val, rel_tol=1e-12, abs_tol=0.0)
    assert math.isclose(result["dx"], dx_val, rel_tol=1e-12, abs_tol=0.0)
    assert math.isclose(result["pdf"], expected_pdf, rel_tol=1e-12, abs_tol=0.0)
    assert result["pdf"] >= 0.0


def test_gb242_callable_cdf_pdf():
    """Test that callables for cdf/pdf are accepted and match float inputs."""
    from math import erf, sqrt, pi, exp

    x = 0.3
    r = 5
    n = 20

    def F(t):
        return 0.5 * (1.0 + erf(t / sqrt(2.0)))

    def f(t):
        return exp(-0.5 * t * t) / sqrt(2.0 * pi)

    fx_val = float(F(x))
    dx_val = float(f(x))

    result_callable = gibbons_order_pdf(x, r, n, F, f)
    result_value = gibbons_order_pdf(x, r, n, fx_val, dx_val)

    assert math.isclose(result_callable["pdf"], result_value["pdf"],
                        rel_tol=1e-12, abs_tol=0.0)
    assert math.isclose(result_callable["coef"], result_value["coef"],
                        rel_tol=1e-12, abs_tol=0.0)


def test_gb242_edge():
    """Test edge cases: r=1 (minimum) and r=n (maximum)."""
    from math import erf, sqrt, pi, exp

    rng = np.random.default_rng(42)
    x = float(rng.normal(0.0, 1.0))
    n = 10
    fx_val = float(0.5 * (1.0 + erf(x / sqrt(2.0))))
    dx_val = float(exp(-0.5 * x * x) / sqrt(2.0 * pi))

    # r = 1 -> only the (1 - F)^(n-1) * f factor survives.
    r = 1
    res_min = gibbons_order_pdf(x, r, n, fx_val, dx_val)
    expected_min = float(
        math.factorial(n) / (math.factorial(0) * math.factorial(n - 1))
        * (1.0 - fx_val) ** (n - 1)
        * dx_val
    )
    assert math.isclose(res_min["pdf"], expected_min,
                        rel_tol=1e-12, abs_tol=0.0)
    assert res_min["r"] == 1
    assert res_min["n"] == n

    # r = n -> only the F^(n-1) * f factor survives.
    r = n
    res_max = gibbons_order_pdf(x, r, n, fx_val, dx_val)
    expected_max = float(
        math.factorial(n) / (math.factorial(n - 1) * math.factorial(0))
        * fx_val ** (n - 1)
        * dx_val
    )
    assert math.isclose(res_max["pdf"], expected_max,
                        rel_tol=1e-12, abs_tol=0.0)
    assert res_max["r"] == n
    assert res_max["n"] == n


def test_gb242_invalid_r_raises():
    """r outside 1..n must raise ValueError."""
    import pytest

    with pytest.raises(ValueError):
        gibbons_order_pdf(0.0, 0, 5, 0.5, 0.4)

    with pytest.raises(ValueError):
        gibbons_order_pdf(0.0, 6, 5, 0.5, 0.4)
