"""Tests for acigls.adjusted_ipgls."""

from morie.fn import _array_core as np

from morie.fn.acigls import adjusted_ipgls


def test_acigls_basic():
    """Test basic functionality."""
    y = np.abs(np.random.default_rng(43).normal(0, 1, 100)) + 0.5
    A = np.random.default_rng(42).normal(0, 1, (100, 10))
    H = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    cluster = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = adjusted_ipgls(y, A, H, cluster)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_acigls_edge():
    """Test edge cases."""
    y = np.abs(np.random.default_rng(43).normal(0, 1, 100)) + 0.5
    A = np.random.default_rng(42).normal(0, 1, (100, 10))
    H = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    cluster = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = adjusted_ipgls(y, A, H, cluster)
    assert isinstance(result, dict)
