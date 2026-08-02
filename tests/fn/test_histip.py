"""Tests for morie.fn.histip — histogram ideal points."""

from morie.fn import _array_core as np

from morie.fn.histip import histip


def test_histip_smoke():
    r = histip(np.arange(100), bins=10)
    assert r.name == "histogram_ideal_points"
    assert r.value == 100
    assert len(r.extra["counts"]) == 10


def test_cheatsheet():
    from morie.fn.histip import cheatsheet

    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
