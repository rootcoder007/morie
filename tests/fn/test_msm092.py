"""Tests for msm092.mvsml_bayesian_regression_pt2_eq_7_3."""

from morie.fn import _array_core as np

from morie.fn.msm092 import mvsml_bayesian_regression_pt2_eq_7_3


def test_msm092_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    X_E = rng.normal(0, 1, (n, 3))
    X = rng.normal(0, 1, (n, 3))
    X_EM = rng.normal(0, 1, (n, 3))
    result = mvsml_bayesian_regression_pt2_eq_7_3(n, X_E=X_E, X=X, X_EM=X_EM)
    assert isinstance(result, dict)
    assert "estimate" in result


def test_msm092_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 5
    X_E = rng.normal(0, 1, (n, 2))
    X = rng.normal(0, 1, (n, 2))
    X_EM = rng.normal(0, 1, (n, 2))
    result = mvsml_bayesian_regression_pt2_eq_7_3(n, X_E=X_E, X=X, X_EM=X_EM)
    assert isinstance(result, dict)
    assert "estimate" in result
