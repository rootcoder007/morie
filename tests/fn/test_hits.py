"""Tests for hits.hits."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.hits import hits


def test_hits_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 10
    A = rng.integers(0, 2, (n, n))
    result = hits(A, 50)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "hubs" in result
    assert "authorities" in result
    assert "top_hub" in result
    assert math.isfinite(result["estimate"])
    assert len(result["hubs"]) == n
    assert len(result["authorities"]) == n
    assert 1 <= result["top_hub"] <= n


def test_hits_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 5
    A = rng.integers(0, 2, (n, n))
    with pytest.raises(ValueError):
        hits(A, 0)
