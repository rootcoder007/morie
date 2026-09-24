"""Tests for line.line."""

from morie.fn import _array_core as np

from morie.fn.line import line


def test_line_basic():
    """Test basic functionality."""
    G = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = line(G)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_line_edge():
    """Test edge cases."""
    G = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = line(G)
    assert isinstance(result, dict)
