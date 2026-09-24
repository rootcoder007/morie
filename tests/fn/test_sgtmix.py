"""Tests for sgtmix.sgt_mixing_time."""

from morie.fn import _array_core as np

from morie.fn.sgtmix import sgt_mixing_time


def test_sgtmix_basic():
    """Test basic functionality."""
    A = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = sgt_mixing_time(A)
    assert isinstance(result, dict)
    assert "estimate" in result or "tau_mix" in result


def test_sgtmix_edge():
    """Test edge cases."""
    A = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = sgt_mixing_time(A)
    assert isinstance(result, dict)
