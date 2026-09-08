"""Test buffr."""

from morie.fn import _array_core as np

from morie.fn.buffr import buffr


def test_buffr_basic():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 30)
    y = rng.uniform(0, 100, 30)
    v = rng.standard_normal(30)
    r = buffr(x=x, y=y, values=v)
    assert r.value is not None


def test_buffr_description():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 30)
    y = rng.uniform(0, 100, 30)
    v = rng.standard_normal(30)
    r = buffr(x=x, y=y, values=v)
    assert r.name
