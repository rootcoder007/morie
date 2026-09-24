"""Tests for htvar1.ht_variance."""

from morie.fn import _array_core as np

from morie.fn.htvar1 import ht_variance


def test_htvar1_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(43)
    y = rng.normal(0, 1, 20)
    pi = np.random.default_rng(42).uniform(0.05, 1, 20)
    pi_ij = np.random.default_rng(42).uniform(0.001, 1, (20, 20))
    result = ht_variance(y, pi, pi_ij)
    assert isinstance(result.payload, dict)
    for key in ("estimate", "variance", "total", "se", "n", "method"):
        assert key in result.payload
    assert result.payload["n"] == 20


def test_htvar1_edge():
    """Test edge case with a single sample."""
    rng = np.random.default_rng(43)
    y = rng.normal(0, 1, 1)
    pi = np.random.default_rng(42).uniform(0.5, 1, 1)
    pi_ij = np.random.default_rng(42).uniform(0.5, 1, (1, 1))
    result = ht_variance(y, pi, pi_ij)
    assert isinstance(result.payload, dict)
    for key in ("estimate", "variance", "total", "se", "n", "method"):
        assert key in result.payload
    assert result.payload["n"] == 1
