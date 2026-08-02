"""Test wqtst."""

from morie.fn import _array_core as np

from morie.fn.wqtst import wqtst


def test_wqtst_basic():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 14, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wqtst(data=data, coords=coords, n=20)
    assert r.value is not None


def test_wqtst_description():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 14, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wqtst(data=data, coords=coords, n=20)
    assert r.name
