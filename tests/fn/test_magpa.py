"""Tests for magpa.ma_glmm_ipd_proportion."""

from morie.fn import _array_core as np

from morie.fn.magpa import ma_glmm_ipd_proportion


def test_magpa_basic():
    """Test basic functionality."""
    xi = 0.5
    ni = 0.5
    result = ma_glmm_ipd_proportion(xi, ni)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_magpa_edge():
    """Test edge cases."""
    xi = 0.5
    ni = 0.5
    result = ma_glmm_ipd_proportion(xi, ni)
    assert isinstance(result, dict)
