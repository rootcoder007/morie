"""Tests for hmpip.geron_pipeline."""

from morie.fn import _array_core as np

from morie.fn.hmpip import geron_pipeline


def test_hmpip_basic():
    """Test basic functionality."""
    X_train = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = geron_pipeline(X_train)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_hmpip_edge():
    """Test edge cases."""
    X_train = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = geron_pipeline(X_train)
    assert isinstance(result, dict)
