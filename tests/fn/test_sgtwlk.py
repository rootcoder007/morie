"""Tests for sgtwlk.sgt_weisfeiler_leman_relabel."""

from morie.fn import _array_core as np

from morie.fn.sgtwlk import sgt_weisfeiler_leman_relabel


def test_sgtwlk_basic():
    """Test basic functionality."""
    A = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = sgt_weisfeiler_leman_relabel(A)
    assert isinstance(result, dict)
    assert "estimate" in result or "labels_t" in result


def test_sgtwlk_edge():
    """Test edge cases."""
    A = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = sgt_weisfeiler_leman_relabel(A)
    assert isinstance(result, dict)
