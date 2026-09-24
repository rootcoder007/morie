"""Tests for nrm.nominal_response_bock."""

from morie.fn import _array_core as np

from morie.fn.nrm import nominal_response_bock


def test_nrm_basic():
    """Test basic functionality."""
    theta = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = nominal_response_bock(theta)
    assert isinstance(result, dict)
    assert "p" in result


def test_nrm_edge():
    """Test edge cases."""
    theta = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = nominal_response_bock(theta)
    assert isinstance(result, dict)
