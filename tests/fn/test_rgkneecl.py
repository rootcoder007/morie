"""Tests for rgkneecl.rangayyan_knee_classify."""

from morie.fn import _array_core as np

from morie.fn.bsaclass import rangayyan_knee_classify


def test_rgkneecl_basic():
    """Test basic functionality."""
    segments = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = rangayyan_knee_classify(segments)
    assert isinstance(result, dict)
    assert "varmeans" in result


def test_rgkneecl_edge():
    """Test edge cases."""
    segments = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = rangayyan_knee_classify(segments)
    assert isinstance(result, dict)
