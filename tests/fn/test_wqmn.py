"""Test wqmn."""

from morie.fn import _array_core as np

from morie.fn.wqmn import wqmn


def test_wqmn_basic():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 14, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wqmn(data=data, coords=coords, n=20)
    assert r.value is not None


def test_wqmn_description():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 14, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wqmn(data=data, coords=coords, n=20)
    assert r.name
