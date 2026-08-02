"""Tests for Wordfish scaling."""

from morie.fn import _array_core as np

from morie.fn.wfish import wfish


def test_wfish_smoke():
    rng = np.random.default_rng(42)
    dtm = rng.poisson(5, size=(8, 20)).astype(float)
    r = wfish(dtm, max_iter=20)
    assert r.name == "wordfish_scaling"
    assert "positions" in r.extra
    assert len(r.extra["positions"]) == 8


def test_cheatsheet():
    from morie.fn.wfish import cheatsheet

    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
