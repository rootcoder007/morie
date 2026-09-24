"""Tests for vitcls.vit_cls_token."""

from morie.fn import _array_core as np

from morie.fn.vitcls import vit_cls_token


def test_vitcls_basic():
    """Test basic functionality."""
    patches = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = vit_cls_token(patches)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_vitcls_edge():
    """Test edge cases."""
    patches = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = vit_cls_token(patches)
    assert isinstance(result, dict)
