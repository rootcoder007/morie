"""Tests for gh_c4_21.ghosal_inv_dp."""

from morie.fn import _array_core as np

from morie.fn.gh_c4_21 import ghosal_inv_dp


def test_gh_c4_21_basic():
    """Test basic functionality."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    x = 3.0
    alpha_x = 1.0
    alpha_total = 2.0
    result = ghosal_inv_dp(x, data, alpha_x, alpha_total)
    assert "estimate" in result
    # Independent computation of the formula:
    # count = 0.5 * sum(1{v <= x} + 1{v >= -x})
    count = sum(0.5 * ((1.0 if v <= x else 0.0)
                       + (1.0 if v >= -x else 0.0)) for v in list(data))
    n = len(list(data))
    expected = (float(alpha_x) + count) / (float(alpha_total) + n)
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    assert np.isclose(result["estimate"], expected)
    assert result["n"] == n
    assert np.isclose(result["symmetrized_count"], count)


def test_gh_c4_21_edge():
    """Test edge cases."""
    data = np.array([42.0])
    result = ghosal_inv_dp(10.0, data, 0.5, 1.0)
    assert result["n"] == 1
    # v=42, x=10: v<=x is False (0), v>=-x is True (1) -> count = 0.5
    expected = (0.5 + 0.5) / (1.0 + 1)
    assert np.isclose(result["estimate"], expected)
