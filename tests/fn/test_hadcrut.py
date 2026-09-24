"""Tests for hadcrut.hadcrut."""

import pytest

from morie.fn import _array_core as np

from morie.fn.hadcrut import hadcrut


def test_hadcrut_basic():
    """Test basic functionality."""
    rng_t = np.random.default_rng(43)
    rng_s = np.random.default_rng(42)
    n_lat, n_lon = 4, 8
    T = rng_t.normal(0, 1, (n_lat, n_lon))
    sst = rng_s.normal(0, 1, (n_lat, n_lon))
    result = hadcrut(T, sst)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_hadcrut_edge():
    """Test that fewer than two latitude bands raises ValueError."""
    rng_t = np.random.default_rng(43)
    rng_s = np.random.default_rng(42)
    n_lat, n_lon = 1, 4
    T = rng_t.normal(0, 1, (n_lat, n_lon))
    sst = rng_s.normal(0, 1, (n_lat, n_lon))
    with pytest.raises(ValueError):
        hadcrut(T, sst)
