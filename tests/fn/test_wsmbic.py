"""Tests for wsmbic.wasserman_bic."""

from morie.fn import _array_core as np

from morie.fn.wsmbic import wasserman_bic


def test_wsmbic_basic():
    """Test basic functionality."""
    loglik = 0.1
    k = 0.1
    n = 5
    result = wasserman_bic(loglik, k, n)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_wsmbic_edge():
    """Test edge cases."""
    loglik = 0.1
    k = 0.1
    n = 5
    result = wasserman_bic(loglik, k, n)
    assert isinstance(result, dict)
