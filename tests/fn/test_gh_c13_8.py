"""Tests for gh_c13_8.ghosal_ntr_def."""

from morie.fn import _array_core as np

from morie.fn.gh_c13_8 import ghosal_ntr_def


def test_gh_c13_8_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_ntr_def(x)
    assert "estimate" in result
    # F(t) = 1 - exp(-M(t)); M = sum of positive increments = 15
    expected_estimate = 1.0 - np.exp(-(1.0 + 2.0 + 3.0 + 4.0 + 5.0))
    assert np.all(np.isclose(np.asarray(result["estimate"], dtype=float),
                             np.asarray(expected_estimate, dtype=float)))
    # F_path should have one entry per increment
    assert len(result["F_path"]) == len(x)
    # F_path must be nondecreasing for nonnegative increments
    assert result["nondecreasing"] is True or result["nondecreasing"]
    # Method tag
    assert "NTR" in result["method"]


def test_gh_c13_8_edge():
    """Test edge cases: single increment."""
    # Single increment: M = 42.0, so F = 1 - exp(-42)
    result = ghosal_ntr_def(np.array([42.0]))
    assert "estimate" in result
    expected_estimate = 1.0 - np.exp(-42.0)
    assert np.all(np.isclose(np.asarray(result["estimate"], dtype=float),
                             np.asarray(expected_estimate, dtype=float)))
    assert len(result["F_path"]) == 1
    assert result["nondecreasing"] is True or result["nondecreasing"]


def test_gh_c13_8_nonnegative_required():
    """Documented constraint: increments must be nonnegative."""
    import pytest
    with pytest.raises(ValueError):
        ghosal_ntr_def(np.array([1.0, -0.1, 2.0]))


def test_gh_c13_8_seed_kwarg():
    """seed kwarg must be accepted."""
    x = np.array([1.0, 2.0, 3.0])
    r1 = ghosal_ntr_def(x, seed=42)
    r2 = ghosal_ntr_def(x, seed=7)
    # Deterministic from increments regardless of seed for this construction
    assert r1["estimate"] == r2["estimate"]
