"""Tests for fzt55.fauzi_thm5_5_bdfree_kde_bv."""

from morie.fn import _array_core as np

from morie.fn.fzt55 import fauzi_thm5_5_bdfree_kde_bv


def test_fzt55_basic():
    """Test basic functionality with documented parameters."""
    n = 100
    h = 0.3
    density = 0.4  # f_X(x)
    c2 = 1.5       # Theorem 5.5 coefficient
    dg = 2.0       # g'(g^{-1}(x)), strictly positive
    mu2 = 1.0
    rk = 1.0 / (2.0 * np.sqrt(np.pi))

    result = fauzi_thm5_5_bdfree_kde_bv(n, h, density, c2, dg, mu2, rk)

    # Documented return keys
    expected_keys = {"bias", "variance", "se", "mse", "hopt", "h", "n", "method"}
    assert set(result.keys()) == expected_keys

    # Independent formulas from Theorem 5.5, Eqs. (5.10)-(5.11)
    expected_bias = h * h * c2 / (2.0 * dg) * mu2
    expected_var = density * rk / (n * h * dg)
    expected_se = np.sqrt(expected_var)
    expected_mse = expected_bias * expected_bias + expected_var
    expected_hopt = (density * rk / (dg * 4.0 * (c2 * mu2 / (2.0 * dg)) ** 2 * n)) ** 0.2

    assert result["bias"] == expected_bias
    assert result["variance"] == expected_var
    assert result["se"] == expected_se
    assert result["mse"] == expected_mse
    assert result["hopt"] == expected_hopt
    assert result["h"] == float(h)
    assert result["n"] == int(n)
    assert result["method"] == "boundary-free KDE bias and variance (Theorem 5.5)"


def test_fzt55_edge():
    """Test edge cases: rk defaults to Gaussian, mu2 defaults to 1.0."""
    n = 50
    h = 0.5
    density = 0.3
    c2 = 2.0
    dg = 1.5

    result = fauzi_thm5_5_bdfree_kde_bv(n, h, density, c2, dg)

    rk_default = 1.0 / (2.0 * np.sqrt(np.pi))
    expected_bias = h * h * c2 / (2.0 * dg) * 1.0
    expected_var = density * rk_default / (n * h * dg)
    expected_mse = expected_bias * expected_bias + expected_var

    assert isinstance(result, dict)
    assert result["bias"] == expected_bias
    assert result["variance"] == expected_var
    assert result["mse"] == expected_mse
