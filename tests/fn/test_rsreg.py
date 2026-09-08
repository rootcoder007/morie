"""Test rsreg."""

from morie.fn import _array_core as np

from morie.fn.rsreg import rsreg


def test_rsreg_basic():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsreg(pixels=pixels, n=40)
    assert r.value is not None


def test_rsreg_description():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsreg(pixels=pixels, n=40)
    assert r.name
