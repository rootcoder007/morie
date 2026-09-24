"""Tests for augmented_kriging_variance.augmented_kriging_variance."""

import math

from morie.fn import _array_core as np

from morie.fn.augmented_kriging_variance import (
    augmented_kriging_variance,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r24e4_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    v_ok = float(rng.uniform(0.5, 4.0))
    e_tau2 = float(rng.uniform(0.1, 1.5))
    result = augmented_kriging_variance(v_ok, e_tau2)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
    assert result["value"] == v_ok + e_tau2
    assert "method" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r24e4_edge():
    """Test edge cases."""
    result = augmented_kriging_variance(0.0, 0.0)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
    assert result["value"] == 0.0
    assert "method" in result
