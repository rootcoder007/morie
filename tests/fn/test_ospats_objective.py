"""Tests for ospats_objective.ospats_objective."""

from morie.fn import _array_core as np

from morie.fn.ospats_objective import (
    ospats_objective,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r13e17_basic():
    """Test basic functionality."""
    per_stratum_sums = 0.5
    n_population = 0.5
    result = ospats_objective(per_stratum_sums, n_population)
    assert isinstance(result, dict)
    assert "value" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r13e17_edge():
    """Test edge cases."""
    per_stratum_sums = 0.5
    n_population = 0.5
    result = ospats_objective(per_stratum_sums, n_population)
    assert isinstance(result, dict)
