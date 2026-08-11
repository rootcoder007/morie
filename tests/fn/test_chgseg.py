"""Tests for chgseg (PELT mean-change specialisation)."""

from morie.fn.chgseg import chgseg, changepoint_segmentation
from morie.fn.pelt import pelt

X1 = [0.1, -0.2, 0.05, 0.3, -0.1, 5.2, 4.9, 5.1, 5.3, 4.8,
      1.9, 2.1, 2.0, 1.8, 2.2]


def test_chgseg_equals_pelt_mean():
    a = chgseg(X1, 3.0)
    b = pelt(X1, "mean", 3.0)
    assert a["changepoints"] == b["changepoints"]
    assert a["objective"] == b["objective"]


def test_chgseg_anchor():
    # changepoint::cpt.mean PELT Manual pen 3.0 -> cpts {5, 10}
    assert chgseg(X1, 3.0)["changepoints"] == [5, 10]


def test_alias():
    assert changepoint_segmentation(X1, 3.0)["changepoints"] == [5, 10]
