"""Tests for catnxt: maximum-information item selection in a CAT under
the four-parameter logistic model, with Sympson-Hetter exposure
multipliers. Information is recomputed from the model by hand."""

import math

import pytest

from morie.fn.catnxt import cat_next_item, catnext

ITEMS = [[1.2, -0.5, 0.0, 1.0],     # 2PL
         [0.8, 0.3, 0.2, 1.0],      # 3PL
         [2.0, 0.1, 0.1, 0.95],     # 4PL, sharp
         [1.0, 1.5, 0.0, 1.0]]


def _info(a, b, c, d, theta, D=1.0):
    e = math.exp(D * a * (theta - b))
    P = c + (d - c) * e / (1 + e)
    dP = D * a * e * (d - c) / (1 + e) ** 2
    return dP * dP / (P * (1 - P))


def test_catnxt_basic():
    """Information is the 4PL formula; the pick is its argmax."""
    theta = 0.2
    r = cat_next_item(ITEMS, theta)
    want = [_info(*it, theta) for it in ITEMS]
    assert list(r["information"]) == pytest.approx(want, rel=1e-12)
    assert r["next_item"] == 1 + max(range(4), key=lambda j: want[j])
    assert r["max_information"] == pytest.approx(max(want), rel=1e-12)
    assert r["J"] == 4 and r["n_available"] == 4
    assert catnext is cat_next_item


def test_2pl_information_at_the_difficulty_is_d2a2_over_4():
    a, b, D = 1.7, 0.4, 1.702
    r = cat_next_item([[a, b, 0.0, 1.0]], b, D=D)
    assert r["information"][0] == pytest.approx(D * D * a * a / 4.0, rel=1e-12)


def test_administered_items_are_skipped_and_exposure_reweights():
    theta = 0.2
    want = [_info(*it, theta) for it in ITEMS]
    best = max(range(4), key=lambda j: want[j])
    r = cat_next_item(ITEMS, theta, administered=[best + 1])
    rest = [j for j in range(4) if j != best]
    assert r["next_item"] == 1 + max(rest, key=lambda j: want[j])
    assert r["n_available"] == 3
    expo = [1.0, 1.0, 0.0, 1.0]
    r2 = cat_next_item(ITEMS, theta, exposure=expo)
    assert list(r2["weighted"]) == pytest.approx(
        [e * w for e, w in zip(expo, want)], rel=1e-12)
    assert r2["next_item"] != 3


def test_ties_break_on_the_lowest_index():
    same = [[1.0, 0.0, 0.0, 1.0]] * 3
    assert cat_next_item(same, 0.0)["next_item"] == 1


def test_catnxt_edge():
    with pytest.raises(ValueError, match="a, b, c, d"):
        cat_next_item([[1.0, 0.0]], 0.0)
