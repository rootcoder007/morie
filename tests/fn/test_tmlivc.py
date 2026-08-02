"""Tests for tmlivc."""

from morie.fn import _array_core as np
import pytest

from morie.fn.tmlivc import tmle_iv


def test_tmlivc_basic():
    rng = np.random.default_rng(42)
    n = 4000
    W = rng.normal(size=(n, 2))
    Z = (rng.random(n) < 0.5).astype(float)
    typ = rng.choice(["c", "a", "n"], size=n, p=[0.6, 0.2, 0.2])
    D = np.where(typ == "a", 1.0, np.where(typ == "n", 0.0, Z))
    y = 2.0 * D * (typ == "c") + W[:, 0] + rng.normal(scale=0.5, size=n)
    out = tmle_iv(y, D, Z, W)
    assert out["late"] == pytest.approx(2.0, abs=0.4)
    assert 0.4 < out["compliance"] < 0.8


def test_tmlivc_edge():
    z = np.zeros(100)
    with pytest.raises(ValueError):
        tmle_iv(np.arange(100.0), z, z)  # no instrument variation
    with pytest.raises(ValueError):
        tmle_iv(np.arange(100.0), np.full(100, 0.5), (np.arange(100) % 2).astype(float))
