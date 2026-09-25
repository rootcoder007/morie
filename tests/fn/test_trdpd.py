"""Tests for trdpd.tree_depth_saturation (NUTS tree-depth diagnostic)."""

import pytest

from morie.fn.trdpd import tree_depth_saturation


def test_trdpd_basic():
    """saturated = #{depth >= max}/n; leapfrog load 2^depth per
    iteration; warn is 1 when any iteration hit the cap."""
    d = [3, 5, 10, 7, 10, 2, 4, 10]
    r = tree_depth_saturation(d, max_depth=10)
    assert r["n_saturated"] == 3.0
    assert r["saturated"] == pytest.approx(3 / 8, abs=1e-15)
    assert r["mean_depth"] == pytest.approx(51 / 8, abs=1e-15)
    lf = [2.0 ** v for v in d]
    assert r["total_leapfrog"] == sum(lf)
    assert r["mean_leapfrog"] == pytest.approx(sum(lf) / 8, abs=1e-12)
    assert r["warn"] == 1.0 and r["max_observed"] == 10


def test_trdpd_edge():
    """No saturation gives warn 0; negative depths, an empty chain and a
    negative cap raise."""
    r = tree_depth_saturation([1, 2, 3], max_depth=10)
    assert r["warn"] == 0.0 and r["saturated"] == 0.0
    with pytest.raises(ValueError):
        tree_depth_saturation([-1, 2])
    with pytest.raises(ValueError):
        tree_depth_saturation([])
    with pytest.raises(ValueError):
        tree_depth_saturation([1, 2], max_depth=-1)
