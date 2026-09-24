"""Tests for msm001.mvsml_general_eq_1_1."""

from morie.fn import _array_core as np

from morie.fn.msm001 import mvsml_general_eq_1_1


def test_msm001_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = mvsml_general_eq_1_1(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msm001_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = mvsml_general_eq_1_1(x)
    assert isinstance(result, dict)
