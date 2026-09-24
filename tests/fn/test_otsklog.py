"""Tests for otsklog.ot_sinkhorn_log."""

from morie.fn import _array_core as np

from morie.fn.otsklog import ot_sinkhorn_log


def test_otsklog_basic():
    """Test basic functionality."""
    a = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    b = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    C = np.random.default_rng(43).normal(0.0, 1.0, (8, 8))
    result = ot_sinkhorn_log(a, b, C)
    assert isinstance(result, dict)
    assert "estimate" in result or "T" in result


def test_otsklog_edge():
    """Test edge cases."""
    a = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    b = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    C = np.random.default_rng(43).normal(0.0, 1.0, (8, 8))
    result = ot_sinkhorn_log(a, b, C)
    assert isinstance(result, dict)
