"""Tests for msm257.mvsml_general_eq_1_222."""

from morie.fn import _array_core as np

from morie.fn.msm257 import mvsml_general_eq_1_222


def test_msm257_basic():
    """Test basic functionality."""
    groups = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_general_eq_1_222(groups)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msm257_edge():
    """Test edge cases."""
    groups = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_general_eq_1_222(groups)
    assert isinstance(result, dict)
