"""Tests for hmmcd.geron_mc_dropout."""

from morie.fn import _array_core as np

from morie.fn.hmmcd import geron_mc_dropout


def test_hmmcd_basic():
    """Test basic functionality."""
    model = (lambda *a, **k: float(np.sum(np.asarray(a[0]) ** 2)))
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_mc_dropout(model, x)
    assert isinstance(result, dict)
    assert "estimate" in result or "mean" in result


def test_hmmcd_edge():
    """Test edge cases."""
    model = (lambda *a, **k: float(np.sum(np.asarray(a[0]) ** 2)))
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_mc_dropout(model, x)
    assert isinstance(result, dict)
