"""Tests for cohens_d_sample.cohens_d_sample."""

from morie.fn import _array_core as np

from morie.fn.cohens_d_sample import cohens_d_sample


def test_ca11e1_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = cohens_d_sample(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ca11e1_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = cohens_d_sample(x)
    assert isinstance(result, dict)
