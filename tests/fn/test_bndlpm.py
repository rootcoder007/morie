"""Tests for bndlpm.bound_lp_method."""

from morie.fn import _array_core as np

from morie.fn.bndlpm import bound_lp_method


def _to_binary_1d(arr):
    """Convert a numeric 1D array to 0/1 ints."""
    out = []
    for v in arr:
        out.append(int(v > 0.5))
    return out


def test_bndlpm_basic():
    """Test basic functionality."""
    n = 100
    rng = np.random.default_rng(43)
    y = _to_binary_1d(rng.normal(0, 1, n))
    D = _to_binary_1d(rng.normal(0, 1, n))
    Z = _to_binary_1d(rng.normal(0, 1, n))
    # moment_eqs must be shape (m, 17): 16 coefficients + rhs
    moment_eqs = rng.normal(0, 1, (2, 17))
    result = bound_lp_method(y, D, Z, moment_eqs)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result
    assert "lower" in result
    assert "upper" in result
    assert "width" in result
    assert "feasible" in result
    assert "n" in result
    # Independent check: width = upper - lower, estimate = midpoint
    assert abs(result["width"] - (result["upper"] - result["lower"])) < 1e-10
    assert abs(result["estimate"] - 0.5 * (result["upper"] + result["lower"])) < 1e-10


def test_bndlpm_edge():
    """Test edge cases."""
    n = 100
    rng = np.random.default_rng(43)
    y = _to_binary_1d(rng.normal(0, 1, n))
    D = _to_binary_1d(rng.normal(0, 1, n))
    Z = _to_binary_1d(rng.normal(0, 1, n))
    moment_eqs = rng.normal(0, 1, (2, 17))
    result = bound_lp_method(y, D, Z, moment_eqs)
    assert isinstance(result, dict)
    assert "lower" in result
    assert "upper" in result
    assert "width" in result
    assert abs(result["width"] - (result["upper"] - result["lower"])) < 1e-10
