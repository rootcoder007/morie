"""Tests for correlation entropy."""

from morie.fn import _array_core as np

from morie.fn.cornt import cornt, correlation_entropy


def test_basic():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 200)
    r = correlation_entropy(x, m_max=5)
    assert r.estimate >= 0


def test_alias():
    assert cornt is correlation_entropy
