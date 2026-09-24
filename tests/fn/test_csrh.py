"""Tests for csrh.cause_specific_hazard."""

from morie.fn import _array_core as np

from morie.fn.csrh import cause_specific_hazard


def test_csrh_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 100
    p = 5
    time = np.linspace(0.1, 10, n)
    event_type = rng.integers(0, 3, n)
    X = rng.normal(0, 1, (n, p))
    result = cause_specific_hazard(time, event_type, X, cause=1)
    assert isinstance(result, dict)
    assert any(k in result for k in ("coefficients", "coef", "estimate", "beta"))


def test_csrh_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 40
    p = 3
    time = np.linspace(0.1, 10, n)
    event_type = rng.integers(0, 2, n)
    X = rng.normal(0, 1, (n, p))
    result = cause_specific_hazard(time, event_type, X, cause=1)
    assert isinstance(result, dict)
