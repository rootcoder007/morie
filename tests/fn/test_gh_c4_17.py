"""Tests for gh_c4_17.ghosal_dp_median."""

from morie.fn import _array_core as np

from morie.fn.gh_c4_17 import ghosal_dp_median


def test_gh_c4_17_basic():
    """Test basic functionality."""
    x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    alpha = 100.0
    result = ghosal_dp_median(x, alpha)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))


def test_gh_c4_17_basic_formula():
    """Verify estimate matches the documented integral via composite-midpoint quadrature."""
    g = 0.4
    alpha = 80.0
    n_grid = 4000
    a, b = alpha * g, alpha * (1.0 - g)
    h = 0.5 / n_grid
    # Composite-midpoint quadrature on [1/2, 1), the documented formula:
    # H(x) = sum_{i=0}^{n_grid-1} be(u; a, b) * h, with u = 1/2 + (i+0.5)*h.
    # We compute the integrand with plain arithmetic using the Beta PDF,
    # B(a,b) = Gamma(a)Gamma(b)/Gamma(a+b), be(u;a,b) = u^{a-1}(1-u)^{b-1}/B(a,b).
    from math import lgamma, exp

    def _beta_pdf(u, a, b):
        if u <= 0.0 or u >= 1.0:
            return 0.0
        log_num = (a - 1.0) * np.log(u) + (b - 1.0) * np.log(1.0 - u)
        log_den = lgamma(a) + lgamma(b) - lgamma(a + b)
        return exp(log_num - log_den)

    expected = 0.0
    for i in range(n_grid):
        u = 0.5 + (i + 0.5) * h
        if u >= 1.0:
            continue
        expected += _beta_pdf(u, a, b) * h

    result = ghosal_dp_median(np.array([g]), alpha, n_grid=n_grid)
    estimate = float(np.asarray(result["estimate"], dtype=float))
    assert abs(estimate - expected) < 1e-6


def test_gh_c4_17_keys():
    """Documented result keys are present."""
    result = ghosal_dp_median(np.array([0.5]), 50.0)
    assert "estimate" in result
    assert "beta_params" in result
    assert "method" in result


def test_gh_c4_17_edge():
    """Edge case: single evaluation, still returns a sane estimate in (0,1)."""
    g_val = 0.5
    alpha = 200.0
    result = ghosal_dp_median(np.array([g_val]), alpha)
    estimate = float(np.asarray(result["estimate"], dtype=float))
    # The integral of a Beta PDF over a sub-interval of (0,1) must lie in (0,1).
    assert 0.0 < estimate < 1.0


def test_gh_c4_17_invalid_g():
    """Documented constraint: 0 < G(x) < 1."""
    import pytest

    with pytest.raises(ValueError):
        ghosal_dp_median(np.array([0.0]), 10.0)
    with pytest.raises(ValueError):
        ghosal_dp_median(np.array([1.0]), 10.0)
