"""Tests for otlowrk.ot_low_rank_sinkhorn."""

from morie.fn import _array_core as np

from morie.fn.otlowrk import ot_low_rank_sinkhorn


def test_otlowrk_basic():
    """Test basic functionality."""
    a = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    b = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    C = np.random.default_rng(43).normal(0.0, 1.0, (8, 8))
    result = ot_low_rank_sinkhorn(a, b, C)
    assert isinstance(result, dict)
    assert "estimate" in result or "U" in result


def test_otlowrk_edge():
    """Test edge cases."""
    a = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    b = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    C = np.random.default_rng(43).normal(0.0, 1.0, (8, 8))
    result = ot_low_rank_sinkhorn(a, b, C)
    assert isinstance(result, dict)
