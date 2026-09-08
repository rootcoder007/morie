"""Tests for morie.fn.imrsz."""

from morie.fn import _array_core as np

from morie.fn.imrsz import imrsz


def test_imrsz_smoke():
    rng = np.random.default_rng(42)
    result = imrsz(image=rng.uniform(size=(32, 32)))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.imrsz import cheatsheet

    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
