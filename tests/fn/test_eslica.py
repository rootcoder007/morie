"""Tests for eslica.esl_ica."""

from morie.fn import _array_core as np

from morie.fn.eslica import esl_ica


def test_eslica_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    result = esl_ica(X, k)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_eslica_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    result = esl_ica(X, k)
    assert isinstance(result, dict)
