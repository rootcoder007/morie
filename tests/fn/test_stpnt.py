"""Tests for morie.fn.stpnt."""

from morie.fn import _array_core as np

from morie.fn.stpnt import stpnt


def test_stpnt_smoke():
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    t = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = stpnt(x, y, t)
    assert result is not None
    assert hasattr(result, "name")
    assert result.statistic is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.stpnt import cheatsheet

    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
