"""Test ppitv."""

from morie.fn import _array_core as np

from morie.fn.ppitv import ppitv


def test_ppitv_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppitv(points=pts, n=30)
    assert r.value is not None


def test_ppitv_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppitv(points=pts, n=30)
    assert r.name
