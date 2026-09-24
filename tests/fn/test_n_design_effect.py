"""Tests for n_design_effect.n_design_effect."""

from morie.fn import _array_core as np

from morie.fn.n_design_effect import (
    n_design_effect,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r12e14_basic():
    """Test basic functionality."""
    design_effect = 0.5
    n_si = 0.5
    result = n_design_effect(design_effect, n_si)
    assert isinstance(result, dict)
    assert "value" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r12e14_edge():
    """Test edge cases."""
    design_effect = 0.5
    n_si = 0.5
    result = n_design_effect(design_effect, n_si)
    assert isinstance(result, dict)
