"""Test turng."""

from morie.fn import _array_core as np

from morie.fn.turng import turng


def test_turng_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = turng(points=pts, n=40)
    assert r.value is not None


def test_turng_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = turng(points=pts, n=40)
    assert r.name
