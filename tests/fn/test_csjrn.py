"""Test csjrn."""

from morie.fn import _array_core as np

from morie.fn.csjrn import csjrn


def test_csjrn_basic():
    rng = np.random.default_rng(42)
    inc = rng.poisson(20, 20)
    pop = rng.poisson(5000, 20) + 100
    r = csjrn(incidents=inc, population=pop, n=20)
    assert r.value is not None


def test_csjrn_description():
    rng = np.random.default_rng(42)
    inc = rng.poisson(20, 20)
    pop = rng.poisson(5000, 20) + 100
    r = csjrn(incidents=inc, population=pop, n=20)
    assert r.name
