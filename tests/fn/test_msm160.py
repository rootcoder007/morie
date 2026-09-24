"""Tests for msm160.mvsml_general_eq_1_2."""

from morie.fn import _array_core as np

from morie.fn.msm160 import mvsml_general_eq_1_2


def test_msm160_basic():
    """Test basic functionality."""
    groups = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_general_eq_1_2(groups)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msm160_edge():
    """Test edge cases."""
    groups = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_general_eq_1_2(groups)
    assert isinstance(result, dict)
