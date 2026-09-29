"""Tests for morie.fn.bbss — Blackbox sum of squares."""

from morie.fn import _array_core as np
from morie.fn.bbss import bbss


def test_bbss_smoke():
    Z = np.array([[1, 2], [3, 4.0]])
    r = bbss(Z)
    assert r.name == "bb_sum_squares"
    assert r.value > 0
    assert "grand_mean" in r.extra


def test_cheatsheet():
    from morie.fn.bbss import cheatsheet

    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0


def test_bbss_total_sum_of_squares_recomputed():
    import pytest

    Z = [[1.0, 4.0, float("nan")], [2.0, 3.0, 5.0], [0.5, 6.0, 2.0]]
    vals = [v for r in Z for v in r if v == v]
    g = sum(vals) / len(vals)
    r = bbss(Z)
    assert r.value == pytest.approx(sum((v - g) ** 2 for v in vals), rel=1e-14)
    assert r.extra["grand_mean"] == pytest.approx(g, rel=1e-14)
    assert r.extra["n_obs"] == 8
