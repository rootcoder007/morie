"""Test rssrf."""

from morie.fn import _array_core as np

from morie.fn.rssrf import rssrf


def test_rssrf_basic():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rssrf(pixels=pixels, n=40)
    assert r.value is not None


def test_rssrf_description():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rssrf(pixels=pixels, n=40)
    assert r.name
