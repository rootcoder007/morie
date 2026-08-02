"""Test sterea."""

from morie.fn import _array_core as np

from morie.fn.sterea import sterea


def test_sterea_basic():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = sterea(coords=coords, n=20)
    assert r.value is not None


def test_sterea_description():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = sterea(coords=coords, n=20)
    assert r.name
