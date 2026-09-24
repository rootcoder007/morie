"""Tests for aitzlr.compositional_zero_lrem."""

from morie.fn import _array_core as np
from morie.fn.aitzlr import compositional_zero_lrem


def test_aitzlr_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    # Compositional data requires strictly positive values for log-ratio transforms
    X = rng.uniform(0.5, 5.0, (40, 3))
    dl = [0.1, 0.2, 0.3]
    n_iter = 10
    result = compositional_zero_lrem(X, dl, n_iter)
    assert hasattr(result, "title")
    assert hasattr(result, "payload")
    assert isinstance(result.payload, dict)
    assert len(result.payload) > 0


def test_aitzlr_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    # Small but strictly positive detection limits; positive compositional data
    X = rng.uniform(0.5, 5.0, (20, 3))
    dl = [0.01, 0.01, 0.01]
    n_iter = 5
    result = compositional_zero_lrem(X, dl, n_iter)
    assert hasattr(result, "title")
    assert hasattr(result, "payload")
    assert isinstance(result.payload, dict)
    assert len(result.payload) > 0
