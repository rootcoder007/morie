"""Tests for km129.kamath_ch9_modality_encoder."""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn.km129 import kamath_ch9_modality_encoder


def test_km129_basic():
    """Test basic functionality with a lambda encoder."""
    I_X = [3.0, 4.0]
    ME_X = lambda z: [z[0], z[1], 0.0]
    result = kamath_ch9_modality_encoder(I_X, ME_X)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "features" in result
    assert "shape" in result
    assert "n" in result
    assert "method" in result
    assert math.isfinite(result["estimate"])
    assert math.isclose(result["estimate"], 5.0)
    assert result["features"] == [3.0, 4.0, 0.0]
    assert result["n"] == 3


def test_km129_edge():
    """Test with array-like input and an encoder that scales features."""
    rng = np.random.default_rng(0)
    I_X = rng.normal(0, 1, 5)
    ME_X = lambda z: [2.0 * float(v) for v in z]
    result = kamath_ch9_modality_encoder(I_X, ME_X)
    assert isinstance(result, dict)
    assert math.isfinite(result["estimate"])
    assert result["estimate"] >= 0.0
    expected = 2.0 * math.sqrt(sum(float(v) ** 2 for v in I_X))
    assert math.isclose(result["estimate"], expected, rel_tol=1e-9)
    assert result["n"] == 5
    assert len(result["features"]) == 5
    assert all(math.isfinite(v) for v in result["features"])
