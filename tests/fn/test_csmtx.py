"""Tests for morie.fn.csmtx."""

from morie.fn import _array_core as np

from morie.fn.csmtx import csmtx


def test_csmtx_smoke():
    rng = np.random.default_rng(42)
    result = csmtx(m=7, n=5)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.csmtx import cheatsheet

    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
