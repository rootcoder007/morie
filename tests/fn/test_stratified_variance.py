"""Tests for stratified_variance.stratified_variance."""

from morie.fn import _array_core as np

from morie.fn.stratified_variance import (
    stratified_variance,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r4e4_basic():
    """Test basic functionality."""
    stratum_variances = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    stratum_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = stratified_variance(stratum_variances, stratum_weights)
    assert isinstance(result, dict)
    assert "value" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r4e4_edge():
    """Test edge cases."""
    stratum_variances = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    stratum_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = stratified_variance(stratum_variances, stratum_weights)
    assert isinstance(result, dict)
