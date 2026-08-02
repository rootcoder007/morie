"""Test ubcng."""

from morie.fn import _array_core as np

from morie.fn.ubcng import ubcng


def test_ubcng_basic():
    rng = np.random.default_rng(42)
    pop = rng.poisson(5000, 20)
    area = rng.uniform(1, 100, 20)
    r = ubcng(population=pop, area=area, n=20)
    assert r.value is not None


def test_ubcng_description():
    rng = np.random.default_rng(42)
    pop = rng.poisson(5000, 20)
    area = rng.uniform(1, 100, 20)
    r = ubcng(population=pop, area=area, n=20)
    assert r.name
