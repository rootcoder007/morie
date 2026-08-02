"""Test pcfsp."""

from morie.fn import _array_core as np

from morie.fn.pcfsp import pcfsp


def test_pcfsp_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = pcfsp(points=pts, n=40)
    assert r.value is not None


def test_pcfsp_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = pcfsp(points=pts, n=40)
    assert r.name
