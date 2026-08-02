"""Test wqas."""

from morie.fn import _array_core as np

from morie.fn.wqas import wqas


def test_wqas_basic():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 14, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wqas(data=data, coords=coords, n=20)
    assert r.value is not None


def test_wqas_description():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 14, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wqas(data=data, coords=coords, n=20)
    assert r.name
