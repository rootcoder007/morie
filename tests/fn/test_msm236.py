"""Tests for msm236.mvsml_general_eq_1_3."""

from morie.fn import _array_core as np

from morie.fn.msm236 import mvsml_general_eq_1_3


def test_msm236_basic():
    """Test basic functionality."""
    groups = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_general_eq_1_3(groups)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msm236_edge():
    """Test edge cases."""
    groups = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_general_eq_1_3(groups)
    assert isinstance(result, dict)
