"""Test ubsch."""

from morie.fn import _array_core as np

from morie.fn.ubsch import ubsch


def test_ubsch_basic():
    rng = np.random.default_rng(42)
    pop = rng.poisson(5000, 20)
    area = rng.uniform(1, 100, 20)
    r = ubsch(population=pop, area=area, n=20)
    assert r.value is not None


def test_ubsch_description():
    rng = np.random.default_rng(42)
    pop = rng.poisson(5000, 20)
    area = rng.uniform(1, 100, 20)
    r = ubsch(population=pop, area=area, n=20)
    assert r.name
