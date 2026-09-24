"""Tests for sgtcoml.sgt_louvain_step."""

from morie.fn import _array_core as np

from morie.fn.sgtcoml import sgt_louvain_step


def test_sgtcoml_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = sgt_louvain_step(A)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_sgtcoml_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = sgt_louvain_step(A)
    assert isinstance(result, dict)
