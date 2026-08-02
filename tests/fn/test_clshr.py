"""Test clshr."""

from morie.fn import _array_core as np

from morie.fn.clshr import clshr


def test_clshr_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clshr(data=data, n=30, k=3)
    assert r.value is not None


def test_clshr_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clshr(data=data, n=30, k=3)
    assert r.name
