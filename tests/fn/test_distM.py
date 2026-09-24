"""Tests for distM.distmult."""

import math

from morie.fn import _array_core as np

from morie.fn.distM import distmult


def test_distM_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    m = 40
    ne = 10
    triples = rng.integers(0, ne, (m, 3))
    dim = 5
    result = distmult(triples, dim)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "scores" in result
    assert "symmetric_gap" in result
    assert "m" in result
    assert "dim" in result
    assert result["m"] == m
    assert result["dim"] == dim
    assert math.isfinite(result["estimate"])
    assert math.isclose(result["symmetric_gap"], 0.0, abs_tol=1e-10)


def test_distM_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    m = 5
    ne = 4
    triples = rng.integers(0, ne, (m, 3))
    dim = 2
    result = distmult(triples, dim)
    assert isinstance(result, dict)
    assert result["m"] == m
    assert result["dim"] == dim
    assert math.isfinite(result["estimate"])
    assert math.isclose(result["symmetric_gap"], 0.0, abs_tol=1e-10)
