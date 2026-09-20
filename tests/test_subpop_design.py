# SPDX-License-Identifier: AGPL-3.0-or-later
"""Expectations are hand-computed from the defining formulas."""

import math

import pytest

from morie.subpop_design import (
    morie_alloc_optimal,
    morie_deff_cluster,
    morie_dif_delta_mh,
    morie_dif_sample_size,
    morie_invariance_compare,
    morie_irt_marginal_reliability,
    morie_irt_theta_se,
    morie_misclass_correct,
    morie_misclass_count,
    morie_neff_cluster,
    morie_oversample_factor,
    morie_rake,
    morie_sample_size_domain,
    morie_sample_size_proportion,
    morie_screen_design,
)

Z975 = 1.9599639845400536


def test_deff_cluster():
    assert morie_deff_cluster(25, 0.02) == pytest.approx(1.48)
    assert morie_deff_cluster(1, 0.9) == 1.0
    assert morie_deff_cluster(10, 0.0) == 1.0
    assert morie_deff_cluster(5, -0.05) < 1.0
    with pytest.raises(ValueError):
        morie_deff_cluster(0.5, 0.1)
    with pytest.raises(ValueError):
        morie_deff_cluster(10, -1)


def test_neff_cluster():
    assert morie_neff_cluster(500, 25, 0.02) == pytest.approx(500 / 1.48)


def test_sample_size_proportion_closed_form():
    r = morie_sample_size_proportion(0.5, 0.05)
    assert r.n_srs == pytest.approx(Z975 ** 2 * 0.25 / 0.05 ** 2)
    assert 384 < r.n_srs < 385
    assert r.n_design == pytest.approx(r.n_srs)
    assert r.n_invite == pytest.approx(r.n_srs)


def test_fpc_deff_and_nonresponse_compose():
    n0 = Z975 ** 2 * 0.25 / 0.05 ** 2
    r = morie_sample_size_proportion(0.5, 0.05, N=1000, deff=1.5,
                                     response_rate=0.6)
    assert r.n_design == pytest.approx((n0 / (1 + (n0 - 1) / 1000)) * 1.5)
    assert r.n_invite == pytest.approx(r.n_design / 0.6)


def test_moe_is_quadratic():
    a = morie_sample_size_proportion(0.5, 0.05).n_srs
    b = morie_sample_size_proportion(0.5, 0.025).n_srs
    assert b / a == pytest.approx(4.0)


def test_half_is_the_conservative_maximum():
    half = morie_sample_size_proportion(0.5, 0.05).n_srs
    for p in (0.1, 0.3, 0.7, 0.9):
        assert morie_sample_size_proportion(p, 0.05).n_srs < half


def test_domain_divides_by_prevalence_and_coverage():
    d = morie_sample_size_domain(0.5, 0.05, domain_prevalence=0.05)
    base = morie_sample_size_proportion(0.5, 0.05)
    assert d.n_domain == pytest.approx(base.n_design)
    assert d.n_overall == pytest.approx(base.n_invite / 0.05)
    assert 7600 < d.n_overall < 7700
    d2 = morie_sample_size_domain(0.5, 0.05, domain_prevalence=0.05,
                                  coverage=0.8)
    assert d2.n_overall == pytest.approx(d.n_overall / 0.8)


def test_oversample_factor():
    o = morie_oversample_factor(0.05, 0.20)
    assert o.factor == pytest.approx(4.75)
    assert o.weight_ratio == pytest.approx(1 / 4.75)
    assert o.deff_weights > 1.0
    flat = morie_oversample_factor(0.2, 0.2)
    assert flat.factor == pytest.approx(1.0)
    assert flat.deff_weights == pytest.approx(1.0)


def test_screen_design():
    s = morie_screen_design(0.05, 400, cost_screen=5, cost_interview=120)
    assert s.n_eligible == pytest.approx(400)
    assert s.n_screen == pytest.approx(8000)
    assert s.cost_total == pytest.approx(88000)
    assert s.cost_per_completed_interview == pytest.approx(220)
    assert s.screening_share_of_cost == pytest.approx(40000 / 88000)


def test_alloc_optimal_reduces_to_neyman():
    N = [8000, 1500, 500]
    S = [1, 1.4, 2]
    a = morie_alloc_optimal(N, S, 900)
    tot = sum(N[i] * S[i] for i in range(3))
    assert a.share == pytest.approx([N[i] * S[i] / tot for i in range(3)])
    assert sum(a.n_h) == pytest.approx(900)
    assert sum(a.n_h_int) == 900
    b = morie_alloc_optimal(N, S, 900, cost_h=[1, 3, 9])
    assert b.share[2] < a.share[2]
    assert b.share[0] > a.share[0]
    c = morie_alloc_optimal(N, S, 900, cost_h=[2, 6, 18])
    assert c.share == pytest.approx(b.share)


def test_rake_hits_both_margins():
    data = {"region": ["north", "north", "south", "south"],
            "group": ["a", "b", "a", "b"]}
    r = morie_rake(data, {"region": {"north": 60, "south": 40},
                          "group": {"a": 70, "b": 30}})
    assert r.converged
    w = r.weights
    assert w[0] + w[1] == pytest.approx(60, abs=1e-6)
    assert w[0] + w[2] == pytest.approx(70, abs=1e-6)
    assert sum(w) == pytest.approx(100, abs=1e-6)


def test_rake_rejects_bad_input():
    data = {"region": ["north", "south"], "group": ["a", "b"]}
    with pytest.raises(ValueError, match="same population total"):
        morie_rake(data, {"region": {"north": 60, "south": 40},
                          "group": {"a": 70, "b": 40}})
    with pytest.raises(ValueError, match="levels absent"):
        morie_rake(data, {"region": {"north": 100}})


def test_rogan_gladen_round_trip():
    assert morie_misclass_correct(0.04, 1, 1).p_corrected == pytest.approx(0.04)
    r = morie_misclass_correct(0.04, 0.75, 0.999)
    assert r.p_corrected == pytest.approx((0.04 + 0.999 - 1) / 0.749)
    assert r.p_corrected > 0.04
    p_true, sens, spec = 0.06, 0.8, 0.99
    p_obs = p_true * sens + (1 - p_true) * (1 - spec)
    assert morie_misclass_correct(p_obs, sens, spec).p_corrected == pytest.approx(p_true)


def test_uninformative_classifier_refused():
    with pytest.raises(ValueError, match="carries no information"):
        morie_misclass_correct(0.04, 0.5, 0.5)


def test_misclass_count():
    r = morie_misclass_count(400, 10000, 0.75, 0.999)
    p = morie_misclass_correct(0.04, 0.75, 0.999).p_corrected
    assert r.count_corrected == pytest.approx(p * 10000)
    assert r.undercount > 0
    with pytest.raises(ValueError):
        morie_misclass_count(400, 100, 0.9, 0.9)


def test_dif_sample_size():
    r = morie_dif_sample_size(0.6, odds_ratio=1.5)
    assert r.p_focal == pytest.approx(2.25 / 3.25)
    big = morie_dif_sample_size(0.6, odds_ratio=2.0).n_focal
    small = morie_dif_sample_size(0.6, odds_ratio=1.2).n_focal
    assert big < small
    unbal = morie_dif_sample_size(0.6, odds_ratio=1.5, ratio=4)
    assert unbal.n_focal < r.n_focal
    assert unbal.n_total > r.n_total
    with pytest.raises(ValueError):
        morie_dif_sample_size(0.6, odds_ratio=1)


def test_delta_mh_and_classification():
    d = morie_dif_delta_mh([1, 1.35, 1.9, 0.5])
    assert [x["magnitude"] for x in d] == ["A", "A", "C", "C"]
    assert [x["favours"] for x in d] == ["neither", "focal", "focal", "reference"]
    assert d[3]["delta_mh"] == pytest.approx(-2.35 * math.log(0.5))
    assert morie_dif_delta_mh(math.exp(-1 / 2.35))[0]["magnitude"] == "B"
    a = morie_dif_delta_mh(1.8)[0]["delta_mh"]
    b = morie_dif_delta_mh(1 / 1.8)[0]["delta_mh"]
    assert a == pytest.approx(-b)


def test_invariance_compare():
    fits = [
        {"model": "configural", "chisq": 120.3, "df": 48, "cfi": 0.981, "rmsea": 0.041},
        {"model": "metric", "chisq": 128.9, "df": 54, "cfi": 0.979, "rmsea": 0.040},
        {"model": "scalar", "chisq": 162.4, "df": 60, "cfi": 0.964, "rmsea": 0.052},
    ]
    r = morie_invariance_compare(fits)
    assert len(r) == 2
    assert r[0]["delta_chisq"] == pytest.approx(8.6)
    assert r[1]["delta_chisq"] == pytest.approx(33.5)
    assert [x["supported"] for x in r] == [True, False]
    with pytest.raises(ValueError, match="least to most constrained"):
        morie_invariance_compare([
            {"model": "a", "chisq": 10, "df": 20, "cfi": 0.99, "rmsea": 0.03},
            {"model": "b", "chisq": 12, "df": 10, "cfi": 0.98, "rmsea": 0.04}])


def test_irt_precision():
    assert morie_irt_theta_se([4, 9, 16]) == pytest.approx([0.5, 1 / 3, 0.25])
    se = [0.3, 0.35, 0.4, 0.5]
    mean_sq = sum(v * v for v in se) / 4
    assert morie_irt_marginal_reliability(se) == pytest.approx(1 - mean_sq)
    assert morie_irt_marginal_reliability(1.0) == pytest.approx(0.0)
