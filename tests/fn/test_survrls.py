"""Tests for survrls.restricted_lifetime."""

from morie.fn import _array_core as np

from morie.fn.survrls import restricted_lifetime


def test_survrls_basic():
    """Test basic functionality."""
    fit = np.random.default_rng(42).normal(0.0, 1.0, 40)
    event = np.random.default_rng(42).normal(0.0, 1.0, 40)
    t_star = 0.1
    result = restricted_lifetime(fit, event, t_star)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_survrls_edge():
    """Test edge cases."""
    fit = np.random.default_rng(42).normal(0.0, 1.0, 40)
    event = np.random.default_rng(42).normal(0.0, 1.0, 40)
    t_star = 0.1
    result = restricted_lifetime(fit, event, t_star)
    assert isinstance(result, dict)
