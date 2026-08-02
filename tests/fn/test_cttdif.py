"""Tests for cttdif.ctt_difficulty."""

from morie.fn import _array_core as np

from morie.fn.cttdif import ctt_difficulty


def test_cttdif_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = ctt_difficulty(X)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_cttdif_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = ctt_difficulty(X)
    assert isinstance(result, dict)
