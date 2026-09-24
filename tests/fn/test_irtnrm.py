"""Tests for irtnrm.nominal_response."""

from morie.fn import _array_core as np

from morie.fn.irtnrm import nominal_response


def test_irtnrm_basic():
    """Test basic functionality."""
    theta = np.random.default_rng(42).normal(0.0, 1.0, 40)
    a = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    c = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = nominal_response(theta, a, c)
    assert isinstance(result, dict)
    assert "p" in result


def test_irtnrm_edge():
    """Test edge cases."""
    theta = np.random.default_rng(42).normal(0.0, 1.0, 40)
    a = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    c = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = nominal_response(theta, a, c)
    assert isinstance(result, dict)
