"""Tests for morie.fn.trnfn."""

from morie.fn import _array_core as np

from morie.fn.trnfn import trnfn


def test_trnfn_smoke():
    num = 0.5
    den = 0.5
    result = trnfn(num, den)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.trnfn import cheatsheet

    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
