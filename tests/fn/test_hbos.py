"""Tests for hbos.hbos."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.hbos import hbos


def test_hbos_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (100, 5))
    result = hbos(X, bins=10)
    assert isinstance(result, dict)
    assert "score" in result
    assert "rank" in result
    assert "densities" in result
    assert "bin_edges" in result
    assert len(result["score"]) == 100
    assert len(result["rank"]) == 100
    assert all(math.isfinite(s) for s in result["score"])


def test_hbos_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (100, 5))
    result = hbos(X, bins=10, mode="dynamic")
    assert isinstance(result, dict)
    assert "score" in result
    assert len(result["score"]) == 100
    with pytest.raises(ValueError):
        hbos(X, bins=0)
    with pytest.raises(ValueError):
        hbos(X, bins=10, mode="invalid")
