"""Tests for wsmlrt.wasserman_lrt."""

from morie.fn import _array_core as np

from morie.fn.wsmlrt import wasserman_lrt


def test_wsmlrt_basic():
    """Test basic functionality."""
    loglik_full = 0.1
    loglik_null = 0.1
    df = 5
    result = wasserman_lrt(loglik_full, loglik_null, df)
    assert isinstance(result, dict)
    assert "statistic" in result or "p_value" in result or "statistic" in result


def test_wsmlrt_edge():
    """Test edge cases."""
    loglik_full = 0.1
    loglik_null = 0.1
    df = 5
    result = wasserman_lrt(loglik_full, loglik_null, df)
    assert isinstance(result, dict)
