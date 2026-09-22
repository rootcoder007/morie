"""Tests for causipw.causal_ipw_truncated."""

from morie.fn import _array_core as np

from morie.fn.causipw import causal_ipw_truncated


def test_causipw_basic():
    """Test basic functionality."""
    treat = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    y = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    ps = np.random.default_rng(42).uniform(0.05, 0.95, 100)
    result = causal_ipw_truncated(treat, y, ps)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_causipw_edge():
    """Test edge cases."""
    treat = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    y = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    ps = np.random.default_rng(42).uniform(0.05, 0.95, 100)
    result = causal_ipw_truncated(treat, y, ps)
    assert isinstance(result, dict)
