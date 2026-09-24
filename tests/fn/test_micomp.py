"""Tests for micomp.mi_compare_models."""

from morie.fn import _array_core as np

from morie.fn.micomp import mi_compare_models


def test_micomp_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    m = 5  # number of imputations
    k = 2  # number of parameters
    theta_list = rng.normal(0, 1, (m, k))
    var_list = [np.eye(k) + rng.normal(0, 0.1, (k, k)) for _ in range(m)]
    result = mi_compare_models(theta_list, var_list)
    assert isinstance(result, dict)
    assert "payload" in result or "statistic" in result


def test_micomp_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    m = 3
    k = 1
    theta_list = rng.normal(0, 1, (m, k))
    var_list = [np.ones((k, k)) for _ in range(m)]
    result = mi_compare_models(theta_list, var_list)
    assert isinstance(result, dict)
