"""Test geodes."""

from morie.fn import _array_core as np

from morie.fn.geodes import geodes


def test_geodes_basic():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = geodes(coords=coords, n=20)
    assert r.value is not None


def test_geodes_description():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = geodes(coords=coords, n=20)
    assert r.name
