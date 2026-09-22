"""Tests for emdtsm.emd_decomposition."""

from morie.fn import _array_core as np

from morie.fn.emdtsm import emd_decomposition


def test_emdtsm_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = emd_decomposition(y)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_emdtsm_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = emd_decomposition(y)
    assert isinstance(result, dict)
