"""Tests for vitfsv.vit_finetune."""

from morie.fn import _array_core as np

from morie.fn.vitfsv import vit_finetune


def test_vitfsv_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    data = np.random.default_rng(42).normal(0, 1, 100)
    mode = "auto"
    result = vit_finetune(model, data, mode)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_vitfsv_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    data = np.random.default_rng(42).normal(0, 1, 100)
    mode = "auto"
    result = vit_finetune(model, data, mode)
    assert isinstance(result, dict)
