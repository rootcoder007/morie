"""Test csenv."""

from morie.fn import _array_core as np

from morie.fn.csenv import csenv


def test_csenv_basic():
    rng = np.random.default_rng(42)
    inc = rng.poisson(20, 20)
    pop = rng.poisson(5000, 20) + 100
    r = csenv(incidents=inc, population=pop, n=20)
    assert r.value is not None


def test_csenv_description():
    rng = np.random.default_rng(42)
    inc = rng.poisson(20, 20)
    pop = rng.poisson(5000, 20) + 100
    r = csenv(incidents=inc, population=pop, n=20)
    assert r.name
