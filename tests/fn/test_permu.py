"""Tests for morie.fn.permu."""

from morie.fn import _array_core as np

from morie.fn.permu import permu


def test_permu_smoke():
    rng = np.random.default_rng(42)
    result = permu(n=5, k=3)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.permu import cheatsheet

    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
