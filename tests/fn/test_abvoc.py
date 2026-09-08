"""Test abvoc."""

from morie.fn import _array_core as np

from morie.fn.abvoc import abvoc


def test_abvoc_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 100, (20, 2))
    r = abvoc(data=data, coords=coords, n=20)
    assert r.value is not None


def test_abvoc_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 100, (20, 2))
    r = abvoc(data=data, coords=coords, n=20)
    assert r.name
