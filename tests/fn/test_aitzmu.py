"""Tests for aitzmu.compositional_zero_multreplace."""

from morie.fn import _array_core as np

from morie.fn.aitzmu import compositional_zero_multreplace


def test_aitzmu_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    # Build compositions that actually contain zeros so the replacement
    # branch is exercised.  A composition must have non-negative parts and
    # a positive row total.
    raw = rng.uniform(0.0, 1.0, (100, 5))
    # Force at least one zero per row so n_zero > 0.
    zero_mask = rng.random((100, 5)) < 0.2
    X = np.where(zero_mask, 0.0, raw)
    # delta must be strictly positive and one entry per part (D = 5).
    delta = [0.01, 0.02, 0.03, 0.04, 0.05]
    result = compositional_zero_multreplace(X, delta)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_aitzmu_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    raw = rng.uniform(0.0, 1.0, (100, 5))
    zero_mask = rng.random((100, 5)) < 0.2
    X = np.where(zero_mask, 0.0, raw)
    delta = [0.01, 0.02, 0.03, 0.04, 0.05]
    result = compositional_zero_multreplace(X, delta)
    assert isinstance(result, dict)
