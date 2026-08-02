"""Test clcph."""

from morie.fn import _array_core as np

from morie.fn.clcph import clcph


def test_clcph_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clcph(data=data, n=30, k=3)
    assert r.value is not None


def test_clcph_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clcph(data=data, n=30, k=3)
    assert r.name
