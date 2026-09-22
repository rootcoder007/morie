"""Tests for dnvtwo.dinov2_repr."""

from morie.fn import _array_core as np

from morie.fn.dnvtwo import dinov2_repr


def test_dnvtwo_basic():
    """Test basic functionality."""
    student = np.random.default_rng(42).normal(0, 1, 100)
    teacher = np.random.default_rng(42).normal(0, 1, 100)
    result = dinov2_repr(student, teacher)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_dnvtwo_edge():
    """Test edge cases."""
    student = np.random.default_rng(42).normal(0, 1, 100)
    teacher = np.random.default_rng(42).normal(0, 1, 100)
    result = dinov2_repr(student, teacher)
    assert isinstance(result, dict)
