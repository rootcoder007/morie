"""Tests for deftef.design_effect."""

from morie.fn import _array_core as np

from morie.fn.deftef import design_effect


def test_deftef_basic():
    """Test basic functionality."""
    design_var = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    srs_var = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = design_effect(design_var, srs_var)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_deftef_edge():
    """Test edge cases."""
    design_var = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    srs_var = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = design_effect(design_var, srs_var)
    assert isinstance(result, dict)
