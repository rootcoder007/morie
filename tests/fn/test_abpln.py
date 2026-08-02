"""Test abpln."""

from morie.fn import _array_core as np

from morie.fn.abpln import abpln


def test_abpln_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 100, (20, 2))
    r = abpln(data=data, coords=coords, n=20)
    assert r.value is not None


def test_abpln_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 100, (20, 2))
    r = abpln(data=data, coords=coords, n=20)
    assert r.name
