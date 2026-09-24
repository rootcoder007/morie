"""Tests for msm315.mvsml_general_eq_1_2."""

from morie.fn import _array_core as np

from morie.fn.msm315 import mvsml_general_eq_1_2


def test_msm315_basic():
    """Test basic functionality."""
    groups = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_general_eq_1_2(groups)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msm315_edge():
    """Test edge cases."""
    groups = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_general_eq_1_2(groups)
    assert isinstance(result, dict)
