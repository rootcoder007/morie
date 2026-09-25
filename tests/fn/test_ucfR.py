"""Tests for ucfR: GroupLens user-based collaborative filtering."""

import math

from morie.fn.ucfR import (co_rated, neighbours, pearson, predict_rating,
                           significance_weight, user_cf)


def _pearson_by_hand(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy)


def test_co_rated_keeps_only_the_items_both_users_rated():
    a = {"i1": 5.0, "i2": 3.0, "i3": 4.0}
    b = {"i2": 2.0, "i3": 5.0, "i4": 1.0}
    c = co_rated(a, b)
    assert c["items"] == ["i2", "i3"]
    assert c["n"] == 2
    assert c["a"] == [3.0, 4.0]
    assert c["b"] == [2.0, 5.0]


def test_significance_weight_is_min_of_n_over_threshold_and_one():
    assert significance_weight(10) == 10.0 / 50.0
    assert significance_weight(80) == 1.0
    assert significance_weight(5, threshold=10) == 0.5


def test_pearson_matches_the_hand_computed_correlation_on_co_rated_items():
    a = {"i1": 1.0, "i2": 2.0, "i3": 3.0, "i4": 9.0}
    b = {"i1": 2.0, "i2": 4.0, "i3": 5.0}
    p = pearson(a, b)
    assert p["n_common"] == 3
    assert p["degenerate"] is False
    assert abs(p["w"] - _pearson_by_hand([1.0, 2.0, 3.0],
                                         [2.0, 4.0, 5.0])) < 1e-12


def test_pearson_significance_scales_the_weight_by_the_overlap():
    a = {"i1": 1.0, "i2": 2.0, "i3": 3.0}
    b = {"i1": 2.0, "i2": 4.0, "i3": 5.0}
    plain = pearson(a, b)["w"]
    scaled = pearson(a, b, significance=True, threshold=10)
    assert scaled["significance_applied"] is True
    assert abs(scaled["w"] - plain * 0.3) < 1e-12


def test_pearson_refuses_a_pair_with_too_little_overlap():
    try:
        pearson({"i1": 1.0, "i2": 2.0}, {"i2": 3.0, "i9": 1.0},
                min_common=2)
    except ValueError as exc:
        assert "co-rated" in str(exc)
    else:
        raise AssertionError("pearson accepted a single co-rated item")


def test_pearson_flags_a_constant_rater_as_degenerate():
    p = pearson({"i1": 3.0, "i2": 3.0, "i3": 3.0},
                {"i1": 1.0, "i2": 2.0, "i3": 5.0})
    assert p["degenerate"] is True
    assert p["w"] == 0.0


def test_neighbours_rank_by_absolute_weight_so_a_disagreer_counts():
    target = {"i1": 1.0, "i2": 2.0, "i3": 3.0}
    others = {
        "agree": {"i1": 2.0, "i2": 3.0, "i3": 2.5},
        "disagree": {"i1": 3.0, "i2": 2.0, "i3": 1.0},
        "flat": {"i1": 2.0, "i2": 2.0, "i3": 2.0},
    }
    out = neighbours(target, others)
    assert out["n"] == 2
    ids = [d["user"] for d in out["neighbours"]]
    assert ids == ["disagree", "agree"]
    assert abs(out["neighbours"][0]["w"] + 1.0) < 1e-12
    assert abs(out["neighbours"][1]["w"] - 0.5) < 1e-12


def test_predict_rating_is_the_own_mean_plus_weighted_deviations():
    target = {"i1": 4.0, "i2": 2.0, "i3": 3.0}
    others = {
        "u1": {"i1": 5.0, "i2": 3.0, "i3": 4.5, "target": 5.0},
        "u2": {"i1": 1.0, "i2": 4.0, "i3": 2.0, "target": 2.0},
    }
    res = predict_rating(target, others, "target")
    mean_a = sum(target.values()) / len(target)
    num = 0.0
    den = 0.0
    for r in others.values():
        w = _pearson_by_hand([target[k] for k in ("i1", "i2", "i3")],
                             [r[k] for k in ("i1", "i2", "i3")])
        mu = sum(r.values()) / len(r)
        num += w * (r["target"] - mu)
        den += abs(w)
    assert res["fell_back"] is False
    assert res["n_neighbours"] == 2
    assert abs(res["user_mean"] - mean_a) < 1e-12
    assert abs(res["estimate"] - (mean_a + num / den)) < 1e-12
    assert res["estimate"] == res["prediction"]


def test_predict_rating_deviations_differ_from_the_naive_raw_mean():
    target = {"i1": 4.0, "i2": 2.0, "i3": 3.0}
    others = {"u1": {"i1": 5.0, "i2": 3.0, "i3": 4.5, "target": 5.0}}
    res = predict_rating(target, others, "target")
    r = others["u1"]
    mu = sum(r.values()) / len(r)
    assert abs(res["naive_weighted_mean"] - 5.0) < 1e-12
    assert abs(res["estimate"] - (3.0 + (5.0 - mu))) < 1e-12


def test_predict_rating_falls_back_to_the_users_own_mean():
    target = {"i1": 4.0, "i2": 2.0, "i3": 3.0}
    others = {"u1": {"i1": 5.0, "i2": 3.0, "i3": 4.5}}
    res = predict_rating(target, others, "never-rated")
    assert res["fell_back"] is True
    assert res["n_neighbours"] == 0
    assert abs(res["estimate"] - 3.0) < 1e-12


def test_predict_rating_refuses_a_target_who_has_rated_nothing():
    try:
        predict_rating({}, {"u1": {"i1": 1.0}}, "i1")
    except ValueError as exc:
        assert "rated nothing" in str(exc)
    else:
        raise AssertionError("predict_rating accepted an empty target")


def test_user_cf_is_the_public_alias_of_predict_rating():
    assert user_cf is predict_rating
