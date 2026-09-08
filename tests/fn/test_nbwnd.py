"""Test nbwnd."""

from morie.fn import _array_core as np

from morie.fn.nbwnd import nbwnd


def test_nbwnd_basic():
    rng = np.random.default_rng(42)
    data = rng.uniform(30, 90, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = nbwnd(data=data, coords=coords, n=20)
    assert r.value is not None


def test_nbwnd_description():
    rng = np.random.default_rng(42)
    data = rng.uniform(30, 90, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = nbwnd(data=data, coords=coords, n=20)
    assert r.name
