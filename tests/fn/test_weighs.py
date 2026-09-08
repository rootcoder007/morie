"""Test weighs."""

from morie.fn import _array_core as np

from morie.fn.weighs import weighs


def test_weighs_basic():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = weighs(coords=coords, n=20)
    assert r.value is not None


def test_weighs_description():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = weighs(coords=coords, n=20)
    assert r.name
