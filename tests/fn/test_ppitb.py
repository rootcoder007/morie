"""Test ppitb."""

from morie.fn import _array_core as np

from morie.fn.ppitb import ppitb


def test_ppitb_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppitb(points=pts, n=30)
    assert r.value is not None


def test_ppitb_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppitb(points=pts, n=30)
    assert r.name
