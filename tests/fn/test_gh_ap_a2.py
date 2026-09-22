"""Tests for gh_ap_a2.ghosal_prohorov_metric."""

from morie.fn import _array_core as np

from morie.fn.gh_ap_a2 import ghosal_prohorov_metric


def test_gh_ap_a2_basic():
    """Test basic functionality."""
    p = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    q = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_prohorov_metric(p, q)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    # Independent computation of expected TV distance from the same inputs.
    p_n = p / sum(p)
    q_n = q / sum(q)
    expected_tv = 0.5 * sum(abs(a - b) for a, b in zip(p_n, q_n))
    assert np.isclose(result["estimate"], expected_tv)
    assert result["upper_bound_by_tv"] is True


def test_gh_ap_a2_edge():
    """Test edge cases."""
    p = np.array([42.0])
    q = np.array([42.0])
    result = ghosal_prohorov_metric(p, q)
    assert "estimate" in result
    assert np.isclose(result["estimate"], 0.0)
