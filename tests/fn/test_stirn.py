"""Tests for morie.fn.stirn."""

from morie.fn import _array_core as np

from morie.fn.stirn import stirn


def test_stirn_smoke():
    rng = np.random.default_rng(42)
    result = stirn(n=5, k=3)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.stirn import cheatsheet

    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
