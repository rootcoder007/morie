"""Test abdst."""

from morie.fn import _array_core as np

from morie.fn.abdst import abdst


def test_abdst_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 100, (20, 2))
    r = abdst(data=data, coords=coords, n=20)
    assert r.value is not None


def test_abdst_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 100, (20, 2))
    r = abdst(data=data, coords=coords, n=20)
    assert r.name
