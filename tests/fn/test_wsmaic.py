"""Tests for wsmaic.wasserman_aic."""

from morie.fn import _array_core as np

from morie.fn.wsmaic import wasserman_aic


def test_wsmaic_basic():
    """Test basic functionality."""
    loglik = 0.1
    k = 0.1
    result = wasserman_aic(loglik, k)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_wsmaic_edge():
    """Test edge cases."""
    loglik = 0.1
    k = 0.1
    result = wasserman_aic(loglik, k)
    assert isinstance(result, dict)
