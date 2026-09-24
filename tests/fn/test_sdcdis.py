"""Tests for sdcdis.spatial_data_distortion."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.sdcdis import spatial_data_distortion


def test_sdcdis_basic():
    """Test basic functionality."""
    n = 40
    rng = np.random.default_rng(42)
    coords = rng.uniform(0.0, 1.0, (n, 2))
    noise_radius = 0.5
    result = spatial_data_distortion(coords, noise_radius, seed=42)
    # result is a RichResult, which behaves like a dict
    assert isinstance(result, dict)
    # estimate: mean realized displacement
    assert "estimate" in result
    estimate = result["estimate"]
    assert math.isfinite(estimate)
    assert estimate >= 0.0
    # masked: displaced coordinates
    assert "masked" in result
    masked = result["masked"]
    # masked should have shape (n, 2)
    assert len(masked) == n
    for row in masked:
        assert len(row) == 2
        assert all(math.isfinite(v) for v in row)


def test_sdcdis_edge():
    """Test edge cases."""
    # empty coords raises ValueError
    with pytest.raises(ValueError):
        spatial_data_distortion([], 0.5)
    # negative noise_radius raises ValueError
    with pytest.raises(ValueError):
        spatial_data_distortion([[0.0, 0.0], [1.0, 1.0]], -0.1)
    # coords with wrong number of columns raises ValueError
    with pytest.raises(ValueError):
        spatial_data_distortion([[0.0, 0.0, 0.0]], 0.5)
