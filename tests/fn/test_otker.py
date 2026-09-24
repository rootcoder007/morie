"""Tests for otker.ot_kernel_emd_approx."""

from morie.fn import _array_core as np

from morie.fn.otker import ot_kernel_emd_approx


def test_otker_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = ot_kernel_emd_approx(X, Y)
    assert isinstance(result, dict)
    assert "EMD_approx" in result


def test_otker_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = ot_kernel_emd_approx(X, Y)
    assert isinstance(result, dict)
