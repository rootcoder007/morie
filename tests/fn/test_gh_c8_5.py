"""Tests for gh_c8_5.ghosal_entropy_cnd."""

from morie.fn import _array_core as np

from morie.fn.gh_c8_5 import ghosal_entropy_cnd


def test_gh_c8_5_basic():
    """Test basic functionality."""
    dim = 5
    radius = 2.0
    n = 10
    eps_n = 0.5
    result = ghosal_entropy_cnd(dim, radius, n, eps_n)
    # Independent computation from the documented formula:
    # log N(eps, ball_d(R), ||.||) <= d * log(3R/eps) (floored at 1.0 inside log).
    expected_estimate = float(dim) * float(np.log(max(3.0 * radius / eps_n, 1.0)))
    expected_bound = float(n) * (float(eps_n) ** 2)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    assert np.isclose(result["estimate"], expected_estimate)
    assert np.isclose(result["bound"], expected_bound)
    assert result["condition_holds"] == (expected_estimate <= expected_bound)
    assert result["method"] == "entropy condition (GvdV 2017 eq. 8.5)"


def test_gh_c8_5_edge():
    """Test edge cases where 3R/eps < 1 forces the log floor."""
    # With eps_n > 3*R, the max(..., 1.0) branch should kick in.
    dim = 3
    radius = 1.0
    n = 100
    eps_n = 10.0  # 3*R/eps_n = 0.3 < 1
    result = ghosal_entropy_cnd(dim, radius, n, eps_n)
    expected_estimate = float(dim) * float(np.log(1.0))  # = 0
    expected_bound = float(n) * (float(eps_n) ** 2)
    assert np.isclose(result["estimate"], expected_estimate)
    assert np.isclose(result["bound"], expected_bound)
    assert result["condition_holds"] == (expected_estimate <= expected_bound)
