"""Tests for gh_c6_10.ghosal_non_iid_con."""

from morie.fn import _array_core as np

from morie.fn.gh_c6_10 import ghosal_non_iid_con


def test_gh_c6_10_basic():
    """Test basic functionality."""
    result = ghosal_non_iid_con()
    assert "estimate" in result
    assert "error_by_n" in result
    assert "avg_kl_at_delta_0.1" in result
    assert "method" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))


def test_gh_c6_10_edge():
    """Test edge cases."""
    result = ghosal_non_iid_con()
    # error_by_n should have one entry per provided n
    assert len(result["error_by_n"]) == 3
    # estimate equals the last entry of error_by_n
    assert result["estimate"] == result["error_by_n"][-1]
    # Independent recomputation of avg_kl_at_delta_0.1 from the documented formula:
    # average over i in 0..29 of 0.01 / (2 * (1 + (i % 3))**2)
    expected_avg_kl = sum(
        0.01 / (2.0 * (1 + (i % 3)) ** 2) for i in range(30)
    ) / 30.0
    assert abs(result["avg_kl_at_delta_0.1"] - expected_avg_kl) < 1e-15
