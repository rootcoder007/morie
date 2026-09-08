"""Test gdmig."""

from morie.fn import _array_core as np

from morie.fn.gdmig import gdmig


def test_gdmig_basic():
    rng = np.random.default_rng(42)
    pop = rng.poisson(10000, 20)
    births = rng.poisson(100, 20)
    deaths = rng.poisson(80, 20)
    r = gdmig(population=pop, births=births, deaths=deaths, n=20)
    assert r.value is not None


def test_gdmig_description():
    rng = np.random.default_rng(42)
    pop = rng.poisson(10000, 20)
    births = rng.poisson(100, 20)
    deaths = rng.poisson(80, 20)
    r = gdmig(population=pop, births=births, deaths=deaths, n=20)
    assert r.name
