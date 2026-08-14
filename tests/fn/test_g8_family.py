"""Tests for the recommender + SVM batch."""
import importlib
import math

import pytest


def M(name):
    return importlib.import_module("morie.fn." + name)


# ---------------------------------------------------------------- svmopt
def test_svmopt_separable_pair_matches_closed_form():
    sv = M("svmopt")
    K = sv.kernel_matrix([[0.0, 0.0], [2.0, 0.0]], "linear")
    r = sv.smo([-1.0, 1.0], K, C=100.0)
    for a in r["alpha"]:
        assert a == pytest.approx(0.5, abs=1e-6)
    assert r["b"] == pytest.approx(-1.0, abs=1e-6)
    assert r["converged"] and abs(r["equality_residual"]) < 1e-9


def test_svmopt_C_bounds_the_multipliers():
    sv = M("svmopt")
    K = sv.kernel_matrix([[0.0, 0.0], [2.0, 0.0]], "linear")
    r = sv.smo([-1.0, 1.0], K, C=0.1)
    assert max(r["alpha"]) == pytest.approx(0.1, abs=1e-9)
    assert all(a <= 0.1 + 1e-9 for a in r["alpha"])


def test_svmopt_clip_bounds_depend_on_label_agreement():
    sv = M("svmopt")
    assert sv._bounds(0, 1, [0.3, 0.4], [1.0, -1.0], 1.0) != \
        sv._bounds(0, 1, [0.3, 0.4], [1.0, 1.0], 1.0)


def test_svmopt_nonseparable_respects_the_constraints():
    sv = M("svmopt")
    K = sv.kernel_matrix([[0.0], [1.0], [2.0]], "linear")
    r = sv.smo([1.0, -1.0, 1.0], K, C=1.0)
    assert all(-1e-9 <= a <= 1.0 + 1e-9 for a in r["alpha"])
    assert abs(r["equality_residual"]) < 1e-8


def test_svmopt_rejects_bad_input():
    sv = M("svmopt")
    K = sv.kernel_matrix([[0.0], [1.0]], "linear")
    with pytest.raises(ValueError):
        sv.smo([1.0, 0.0], K)
    with pytest.raises(ValueError):
        sv.smo([1.0, -1.0], K, C=0.0)
    with pytest.raises(ValueError):
        sv.kernel_matrix([[0.0], [1.0]], "quartic")


# ----------------------------------------------------------------- svdpp
Y2 = {0: [1.0, 0.0], 1: [1.0, 0.0], 2: [0.0, 1.0]}


def test_svdpp_normalisation_exponent():
    sp = M("svdpp")
    assert sp.implicit_term([0], Y2)["scale"] == pytest.approx(1.0)
    assert sp.implicit_term([0, 1, 0, 1], Y2)["scale"] == \
        pytest.approx(0.5)
    assert sp.implicit_term([0, 1, 0, 1], Y2,
                            exponent=0.0)["term"][0] == 4.0
    assert sp.implicit_term([0, 1, 0, 1], Y2,
                            exponent=-1.0)["term"][0] == \
        pytest.approx(1.0)


def test_svdpp_prediction_places_the_term_inside():
    sp = M("svdpp")
    r = sp.predict(3.0, 0.2, -0.1, [1.0, 0.0], [1.0, 0.0], [0, 1],
                   Y2)
    assert r["prediction"] == pytest.approx(
        3.0 + 0.2 - 0.1 + 1.0 + 2.0 / math.sqrt(2.0))


def test_svdpp_handles_a_user_with_no_ratings():
    sp = M("svdpp")
    assert sp.implicit_term([], Y2)["n_rated"] == 0
    r = sp.predict(3.0, 0.0, 0.0, [1.0, 0.0], [1.0, 0.0], [], {})
    assert r["prediction"] == pytest.approx(4.0)


def test_svdpp_training_reduces_error_and_differs_from_plain_svd():
    sp = M("svdpp")
    ratings = [(0, 0, 5.0), (0, 1, 4.0), (1, 0, 4.0), (1, 2, 2.0),
               (2, 1, 5.0), (2, 2, 1.0), (0, 2, 2.0), (1, 1, 4.0)]
    a = sp.fit_svdpp(ratings, 3, 3, factors=2, epochs=60, seed=1)
    b = sp.fit_svdpp(ratings, 3, 3, factors=2, epochs=60, seed=1,
                     implicit=False)
    assert a["rmse"] < a["rmse_history"][0]
    assert abs(a["rmse"] - b["rmse"]) > 1e-6
    assert b["Y"] is None


# ---------------------------------------------------------------- timeRS
def test_timeRS_deviation_is_signed_and_concave():
    tr = M("timeRS")
    assert tr.deviation(110, 100) > 0
    assert tr.deviation(90, 100) < 0
    assert tr.deviation(100, 100) == 0.0
    d1, d4 = tr.deviation(200, 100), tr.deviation(500, 100)
    assert d4 < 4.0 * d1
    assert d1 == pytest.approx(100.0 ** 0.4)


def test_timeRS_beta_is_the_published_value():
    tr = M("timeRS")
    assert tr.BETA == pytest.approx(0.4)
    with pytest.raises(ValueError):
        tr.deviation(110, 100, beta=0.0)


def test_timeRS_bins_are_slow():
    tr = M("timeRS")
    assert tr.time_bin(0) == 0 and tr.time_bin(69) == 0
    assert tr.time_bin(70) == 1
    with pytest.raises(ValueError):
        tr.time_bin(10, bin_days=0)


def test_timeRS_per_day_term_is_separate():
    tr = M("timeRS")
    r = tr.user_bias(0.1, 0.01, 200, 100, {200: 0.5})
    assert r["per_day"] == pytest.approx(0.5)
    assert r["bias"] == pytest.approx(0.1 + 0.01 * r["deviation"]
                                      + 0.5)


def test_timeRS_fit_detects_drift_and_keeps_every_instance():
    tr = M("timeRS")
    rt = [(0, 0, float(t), 3.0 + 0.002 * t)
          for t in range(0, 400, 20)]
    rt += [(1, 1, float(t), 4.0) for t in range(0, 400, 20)]
    f = tr.fit_time_bias(rt, 2, 2, epochs=120, lr=0.02)
    assert abs(f["alpha_user"][0]) > abs(f["alpha_user"][1])
    assert f["n_instances"] == len(rt)


# ------------------------------------------------------------------ ucfR
TARGET = {"a": 3.0, "b": 4.0}
GENEROUS = {"a": 4.0, "b": 5.0, "c": 5.0}


def test_ucfR_correlation_is_scale_free():
    uc = M("ucfR")
    assert uc.pearson(TARGET, GENEROUS)["w"] == pytest.approx(1.0)


def test_ucfR_prediction_uses_deviations():
    uc = M("ucfR")
    r = uc.predict_rating(TARGET, {"g": GENEROUS}, "c")
    assert r["prediction"] == pytest.approx(3.5 + (5.0 - 14.0 / 3.0))
    assert r["naive_weighted_mean"] > r["prediction"]


def test_ucfR_unrated_is_silent_not_zero():
    uc = M("ucfR")
    assert uc.co_rated(TARGET, GENEROUS)["items"] == ["a", "b"]


def test_ucfR_thin_overlap_refused_and_downweighted():
    uc = M("ucfR")
    with pytest.raises(ValueError):
        uc.pearson({"a": 1.0}, {"a": 2.0, "b": 3.0}, min_common=2)
    assert uc.significance_weight(2) == pytest.approx(0.04)
    assert uc.significance_weight(50) == pytest.approx(1.0)


def test_ucfR_falls_back_to_the_user_mean():
    uc = M("ucfR")
    r = uc.predict_rating(TARGET, {"g": {"a": 4.0, "b": 5.0}}, "zz")
    assert r["fell_back"] and r["prediction"] == pytest.approx(3.5)
    with pytest.raises(ValueError):
        uc.predict_rating({}, {"g": GENEROUS}, "a")


# ----------------------------------------------------------------- hybRC
def test_hybRC_order_sensitivity_table():
    hb = M("hybRC")
    for m in ("weighted", "switching", "mixed",
              "feature_combination"):
        assert not hb.is_order_sensitive(m)["order_sensitive"]
    for m in ("cascade", "feature_augmentation", "meta_level"):
        assert hb.is_order_sensitive(m)["order_sensitive"]
    with pytest.raises(ValueError):
        hb.is_order_sensitive("bayesian")


def test_hybRC_weighted_is_symmetric_cascade_is_not():
    hb = M("hybRC")
    A = {"x": 1.0, "y": 0.0}
    B = {"x": 0.0, "y": 1.0}
    assert hb.weighted([A, B])["scores"] == hb.weighted([B, A])["scores"]
    c1 = hb.cascade({"x": 1.0, "y": 1.0, "z": 0.0},
                    {"x": 0.0, "y": 1.0})
    c2 = hb.cascade({"x": 0.0, "y": 1.0},
                    {"x": 1.0, "y": 1.0, "z": 0.0})
    assert c1["ranking"] != c2["ranking"]
    assert c1["primary_respected"]


def test_hybRC_weighted_reports_partial_coverage():
    hb = M("hybRC")
    r = hb.weighted([{"x": 1.0}, {"x": 1.0, "y": 10.0}])
    assert r["partially_scored"] == ["y"]


def test_hybRC_mixed_and_feature_combination():
    hb = M("hybRC")
    mx = hb.mixed([["a", "b"], ["c"]])
    assert [d["item"] for d in mx["presented"]] == ["a", "c", "b"]
    fc = hb.feature_combination([[1.0]], [[2.0, 3.0]])
    assert fc["features"] == [[1.0, 2.0, 3.0]]
    with pytest.raises(ValueError):
        hb.feature_combination([[1.0]], [[2.0], [3.0]])


# ----------------------------------------------------------------- tagRC
def _hub_triples():
    t = [("u1", "t1", "r1"), ("u1", "t1", "r2")]
    for n in range(2, 12):
        t += [("u%d" % n, "tHUB", "r%d" % n),
              ("u%d" % n, "tHUB", "r1")]
    return t


def test_tagRC_graph_is_undirected():
    tg = M("tagRC")
    g = tg.tripartite_graph(_hub_triples())
    for a in g["adjacency"]:
        for b in g["adjacency"][a]:
            assert a in g["adjacency"][b]


def test_tagRC_difference_removes_the_hub():
    tg = M("tagRC")
    fr = tg.folkrank(_hub_triples(), ["t:t1"], weight=0.5)
    assert fr["baseline_ranking"][0] == "t:tHUB"
    assert fr["ranking"].index("t:tHUB") > \
        fr["undifferenced_ranking"].index("t:tHUB") + 10
    assert fr["ranking"][0] == "t:t1"
    assert fr["difference"]["t:tHUB"] < 0.0 < \
        fr["difference"]["t:t1"]


def test_tagRC_preference_vector_has_unit_mass():
    tg = M("tagRC")
    g = tg.tripartite_graph(_hub_triples())
    assert tg.preference_vector(g["nodes"], ["t:t1"])["mass"] == \
        pytest.approx(1.0)
    with pytest.raises(ValueError):
        tg.preference_vector(g["nodes"], ["t:absent"])
    with pytest.raises(ValueError):
        tg.preference_vector(g["nodes"], ["t:t1"], weight=1.5)


# ----------------------------------------------------------------- warpL
def test_warpL_alpha_schemes():
    wp = M("warpL")
    rec = wp.alpha_weights(5, "reciprocal")
    uni = wp.alpha_weights(5, "uniform")
    assert rec[0] == pytest.approx(1.0)
    assert rec[4] == pytest.approx(0.2)
    assert all(rec[i] >= rec[i + 1] for i in range(4))
    assert wp.rank_weight(3, uni) == pytest.approx(3.0)
    with pytest.raises(ValueError):
        wp.alpha_weights(5, "increasing")


def test_warpL_top_weighting_beats_uniform_at_rank_one():
    wp = M("warpL")
    rec = wp.alpha_weights(5, "reciprocal")
    uni = wp.alpha_weights(5, "uniform")
    assert (wp.rank_weight(1, rec) / wp.rank_weight(5, rec)) > \
        (wp.rank_weight(1, uni) / wp.rank_weight(5, uni))


def test_warpL_rank_estimate_is_floor_of_the_ratio():
    wp = M("warpL")
    assert wp.estimate_rank(1, 101) == 100
    assert wp.estimate_rank(100, 101) == 1
    with pytest.raises(ValueError):
        wp.estimate_rank(0, 101)


def test_warpL_sampling_reports_the_cap():
    wp = M("warpL")
    rng = M("_array_core").random.default_rng(3)
    easy = wp.sample_violation(0.0, lambda j: 5.0, 101, rng)
    hard = wp.sample_violation(100.0, lambda j: -5.0, 101, rng)
    assert easy["violated"] and easy["draws"] == 1
    assert easy["estimated_rank"] == 100
    assert not hard["violated"] and hard["capped"]


def test_warpL_step_updates_only_on_a_violation():
    wp = M("warpL")
    rng = M("_array_core").random.default_rng(3)
    rec = wp.alpha_weights(3, "reciprocal")
    hit = wp.warp_step([1.0, 0.0], [[0.9, 0.0], [0.8, 0.0]],
                       [1.0, 0.0], rng, rec)
    assert hit["updated"] and hit["loss"] > 0.0
    miss = wp.warp_step([10.0, 0.0], [[-5.0, 0.0], [-5.0, 0.0]],
                        [1.0, 0.0], rng, rec)
    assert not miss["updated"] and miss["loss"] == 0.0
