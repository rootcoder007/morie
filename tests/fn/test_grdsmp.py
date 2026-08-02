"""Test grdsmp."""

from morie.fn import _array_core as np

from morie.fn.grdsmp import grdsmp


def test_grdsmp_basic():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = grdsmp(coords=coords, n=20)
    assert r.value is not None


def test_grdsmp_description():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = grdsmp(coords=coords, n=20)
    assert r.name
