"""Tests for gh_c8_7.ghosal_fin_apx_pri."""

from morie.fn import _array_core as np

from morie.fn.gh_c8_7 import ghosal_fin_apx_pri


def test_gh_c8_7_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    n = 100
    a = 1.0
    result = ghosal_fin_apx_pri(a, n)
    expected_estimate = float(n) ** (-a / (2.0 * a + 1.0))
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value
    assert np.allclose(np.asarray(result["estimate"], dtype=float), expected_estimate)
    expected_net_size_log = expected_estimate ** (-1.0 / a)
    expected_balance_gap = abs(expected_net_size_log - float(n) * expected_estimate ** 2) / (float(n) * expected_estimate ** 2)
    assert np.allclose(np.asarray(result["net_size_log"], dtype=float), expected_net_size_log)
    assert np.allclose(np.asarray(result["balance_gap"], dtype=float), expected_balance_gap)
    assert "method" in result
    assert result["method"] == "net-prior rate (GvdV 2017 Thm 8.15, Ex 8.16)"


def test_gh_c8_7_edge():
    """Test edge cases."""
    n = 42
    result = ghosal_fin_apx_pri(np.array([42.0]), n)
    a = 42.0
    expected_estimate = float(n) ** (-a / (2.0 * a + 1.0))
    assert np.allclose(np.asarray(result["estimate"], dtype=float), expected_estimate)
