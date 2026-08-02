"""Test wltrn."""

from morie.fn import _array_core as np

from morie.fn.wltrn import wltrn


def test_wltrn_basic():
    rng = np.random.default_rng(42)
    abund = rng.poisson(10, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wltrn(abundance=abund, coords=coords, n=20)
    assert r.value is not None


def test_wltrn_description():
    rng = np.random.default_rng(42)
    abund = rng.poisson(10, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wltrn(abundance=abund, coords=coords, n=20)
    assert r.name
