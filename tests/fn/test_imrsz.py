"""Tests for morie.fn.imrsz."""

from morie.fn import _array_core as np

from morie.fn.imrsz import imrsz


def test_imrsz_smoke():
    image = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = imrsz(image)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.imrsz import cheatsheet

    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
