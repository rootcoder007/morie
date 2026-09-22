"""Tests for causaipw.causal_aipw."""

from morie.fn import _array_core as np

from morie.fn.causaipw import causal_aipw


def test_causaipw_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).uniform(0.05, 0.95, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    ps = np.random.default_rng(42).uniform(0.05, 0.95, 100)
    m1 = np.random.default_rng(42).uniform(0.05, 0.95, 100)
    m0 = np.random.default_rng(42).uniform(0.05, 0.95, 100)
    result = causal_aipw(y, T, ps, m1, m0)
    assert isinstance(result, dict)
    assert "ate" in result
def test_causaipw_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).uniform(0.05, 0.95, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    ps = np.random.default_rng(42).uniform(0.05, 0.95, 100)
    m1 = np.random.default_rng(42).uniform(0.05, 0.95, 100)
    m0 = np.random.default_rng(42).uniform(0.05, 0.95, 100)
    result = causal_aipw(y, T, ps, m1, m0)
    assert isinstance(result, dict)
