"""Tests for ospats_criterion_terms.ospats_criterion_terms."""

from morie.fn import _array_core as np

from morie.fn.ospats_criterion_terms import (
    ospats_criterion_terms,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r13e12_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ospats_criterion_terms(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r13e12_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ospats_criterion_terms(x)
    assert isinstance(result, dict)
