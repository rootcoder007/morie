"""Tests for km138.kamath_ch9_simvlm_mlm."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.km138 import kamath_ch9_simvlm_mlm


def test_km138_basic():
    """Test basic functionality with None theta (x holds probabilities)."""
    x = [0.5, 1.0, 0.25]
    v = [[0.0]]
    x_m = [0, 2]
    result = kamath_ch9_simvlm_mlm(None, x, v, x_m)
    assert isinstance(result, dict)
    assert "estimate" in result
    expected = (math.log(2) + math.log(4)) / 2
    assert math.isfinite(result["estimate"])
    assert abs(result["estimate"] - expected) < 1e-12
    assert result["n_image_regions"] == 1
    assert result["n"] == 3
    assert "method" in result


def test_km138_edge():
    """Test edge case: empty image regions raises ValueError."""
    x = [0.5, 1.0, 0.25]
    v = np.zeros((0, 3))
    x_m = [0, 2]
    with pytest.raises(ValueError):
        kamath_ch9_simvlm_mlm(None, x, v, x_m)
