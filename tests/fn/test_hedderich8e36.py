"""Verification tests for hedderich8e36.

Hedderich, eq (8.36) -- the prediction interval for a single observation. Every expected value is recomputed in the test body
from the equation the module cites.
"""

import math

import pytest

from morie.fn.hedderich8e36 import hedderich_chapter_8_equation_36


X = [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 6.0]]
Y = [2.0, 3.0, 5.0, 4.0, 7.0]
X0 = [1.0, 5.0]


def _fit_by_hand():
    n = len(X)
    sx = sum(r[1] for r in X)
    sxx = sum(r[1] ** 2 for r in X)
    sy = sum(Y)
    sxy = sum(r[1] * v for r, v in zip(X, Y))
    det = n * sxx - sx * sx
    # (X'X)^-1 for a 2 by 2, written out
    inv = [[sxx / det, -sx / det], [-sx / det, n / det]]
    b0 = inv[0][0] * sy + inv[0][1] * sxy
    b1 = inv[1][0] * sy + inv[1][1] * sxy
    rss = sum((v - b0 - b1 * r[1]) ** 2 for r, v in zip(X, Y))
    sigma = math.sqrt(rss / (n - 2))
    lev = sum(X0[a] * inv[a][b] * X0[b] for a in range(2) for b in range(2))
    return b0, b1, sigma, lev


def test_prediction_interval_fit_sigma_and_leverage():
    b0, b1, sigma, lev = _fit_by_hand()
    res = hedderich_chapter_8_equation_36(X, Y, X0)
    assert res["fit"] == pytest.approx(b0 + b1 * X0[1], rel=1e-10)
    assert res["sigma"] == pytest.approx(sigma, rel=1e-10)
    assert res["leverage"] == pytest.approx(lev, rel=1e-10)
    assert res["df"] == len(X) - 2
    assert res["coef"][0] == pytest.approx(b0, rel=1e-10)
    assert res["coef"][1] == pytest.approx(b1, rel=1e-10)


def test_interval_is_symmetric_about_the_fit():
    res = hedderich_chapter_8_equation_36(X, Y, X0)
    assert res["fit"] - res["lower"] == pytest.approx(res["upper"] - res["fit"], rel=1e-12)


def test_single_observation_interval_is_wider_than_the_mean_interval():
    # (8.36) carries sqrt(1 + h), (8.37) only sqrt(h), so their
    # half-widths are in the ratio sqrt((1 + h)/h) at the same level
    _, _, _, lev = _fit_by_hand()
    single = hedderich_chapter_8_equation_36(X, Y, X0, mean=False)
    mean = hedderich_chapter_8_equation_36(X, Y, X0, mean=True)
    ratio = (single["upper"] - single["fit"]) / (mean["upper"] - mean["fit"])
    assert ratio == pytest.approx(math.sqrt((1.0 + lev) / lev), rel=1e-10)


def test_a_higher_level_widens_the_interval():
    narrow = hedderich_chapter_8_equation_36(X, Y, X0, level=0.80)
    wide = hedderich_chapter_8_equation_36(X, Y, X0, level=0.99)
    assert wide["upper"] - wide["fit"] > narrow["upper"] - narrow["fit"]


def test_rejects_a_new_row_of_the_wrong_width():
    with pytest.raises(ValueError):
        hedderich_chapter_8_equation_36(X, Y, [1.0, 5.0, 9.0])
