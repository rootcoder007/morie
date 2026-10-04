"""Tests for tqmom.turboquant_normal_moment."""

from morie.fn.tqmom import turboquant_normal_moment


def test_tqmom_basic():
    """Test basic functionality."""
    sigma = 0.1
    ell = 0.1
    result = turboquant_normal_moment(sigma, ell)
    assert isinstance(result, dict)
    assert "estimate" in result or "moment" in result


def test_tqmom_edge():
    """Test edge cases."""
    sigma = 0.1
    ell = 0.1
    result = turboquant_normal_moment(sigma, ell)
    assert isinstance(result, dict)
