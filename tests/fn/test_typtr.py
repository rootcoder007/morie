"""Tests for morie.fn.typtr."""

from morie.fn import _array_core as np

from morie.fn.typtr import typtr


def test_typtr_smoke():
    rng = np.random.default_rng(42)
    result = typtr(text="The quick brown fox jumps over the lazy dog")
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.typtr import cheatsheet

    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
