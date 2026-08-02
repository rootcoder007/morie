"""Tests for spatiotemporal_methods_in_environmental_epidemiology_with_r_6u35.spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_6_unnumbered_35."""

from morie.fn import _array_core as np

from morie.fn.spatiotemporal_methods_in_environmental_epidemiology_with_r_6u35 import (
    spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_6_unnumbered_35,
)


def test_spatiotemporal_methods_in_environmental_epidemiology_with_r_6u35_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_6_unnumbered_35(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_spatiotemporal_methods_in_environmental_epidemiology_with_r_6u35_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_6_unnumbered_35(x)
    assert isinstance(result, dict)
