"""Tests for hbvMod.hbv_hydrology."""

from morie.fn import _array_core as np

from morie.fn.hbvMod import hbv_hydrology


def test_hbvMod_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    P = rng.normal(0, 1, 100)
    T = rng.integers(0, 2, 100)
    PET = rng.normal(0, 1, 100)
    params = {
        "tt": 0.0,
        "cfmax": 3.0,
        "fc": 100.0,
        "lp": 0.5,
        "beta": 1.0,
        "k0": 0.1,
        "k1": 0.05,
        "k2": 0.01,
        "uzl": 5.0,
        "perc": 1.0,
        "maxbas": 3,
    }
    result = hbv_hydrology(P, T, PET, params)
    assert isinstance(result, dict)


def test_hbvMod_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    P = rng.normal(0, 1, 100)
    T = rng.integers(0, 2, 100)
    PET = rng.normal(0, 1, 100)
    params = {
        "tt": 0.0,
        "cfmax": 3.0,
        "fc": 100.0,
        "lp": 0.5,
        "beta": 1.0,
        "k0": 0.1,
        "k1": 0.05,
        "k2": 0.01,
        "uzl": 5.0,
        "perc": 1.0,
        "maxbas": 3,
    }
    result = hbv_hydrology(P, T, PET, params)
    assert isinstance(result, dict)
