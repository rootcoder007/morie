"""Test sesmr."""

from morie.fn import _array_core as np

from morie.fn.sesmr import sesmr


def test_sesmr_basic():
    rng = np.random.default_rng(42)
    cases = rng.poisson(5, 20)
    pop = rng.poisson(1000, 20) + 100
    coords = rng.uniform(0, 100, (20, 2))
    r = sesmr(cases=cases, population=pop, coords=coords, n=20)
    assert r.value is not None


def test_sesmr_description():
    rng = np.random.default_rng(42)
    cases = rng.poisson(5, 20)
    pop = rng.poisson(1000, 20) + 100
    coords = rng.uniform(0, 100, (20, 2))
    r = sesmr(cases=cases, population=pop, coords=coords, n=20)
    assert r.name
