"""Test sawnr."""

from morie.fn import _array_core as np

from morie.fn.sawnr import sawnr


def test_sawnr_basic():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sawnr(values=vals, n=25)
    assert r.value is not None


def test_sawnr_description():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sawnr(values=vals, n=25)
    assert r.name
