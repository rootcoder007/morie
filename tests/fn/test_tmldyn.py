"""Tests for tmldyn -- CV-TMLE under the optimal dynamic treatment rule.

Replaces a generated test that called a stub returning mean(y). The
truth here is a closed form derived from the structural equations that
generate the data, not another estimator. Full anchor, including the
coverage run and the confounding comparison:
ledger/wave3/anchor_tmldyn.py.
"""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn import _s03core as k
from morie.fn.tmldyn import (exceptional_law_share,
                             intervention_mechanism, sequential_blips,
                             tmle_dynamic_regime)

N = 3000
S_W1 = 0.8
CUT = 0.9 / 0.8


def expit(z):
    return 1.0 / (1.0 + math.exp(-z))


def _phi(z):
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _mu1(a0, w0):
    return 0.5 * w0 + 1.2 * a0


def _B(a0, w0):
    """E[max(0, 0.9 - 0.8*W1) | a0, w0], the value the stage-2 rule adds."""
    m = _mu1(a0, w0)
    z = (CUT - m) / S_W1
    return 0.8 * ((CUT - m) * k.pnorm(z) + S_W1 * _phi(z))


def _v_dyn(a0, w0):
    return (1.0 + 0.5 * w0 + a0 * (0.9 * w0 - 0.1)
            + 0.4 * _mu1(a0, w0) + _B(a0, w0))


def _v_static(a0, a1, w0):
    m = _mu1(a0, w0)
    return (1.0 + 0.5 * w0 + a0 * (0.9 * w0 - 0.1) + 0.4 * m
            + a1 * (0.9 - 0.8 * m))


@pytest.fixture(scope="module")
def two_stage():
    """A design whose optimal rule is dynamic at BOTH stages: the
    stage-2 blip changes sign in W1, and the stage-1 blip changes sign
    in W0 only once the stage-2 rule is carried into it."""
    rng = np.random.default_rng(11)
    W0 = [rng.standard_normal() for _ in range(N)]
    A0 = [1.0 if rng.uniform() < expit(0.8 * W0[i]) else 0.0
          for i in range(N)]
    W1 = [0.5 * W0[i] + 1.2 * A0[i] + S_W1 * rng.standard_normal()
          for i in range(N)]
    A1 = [1.0 if rng.uniform() < expit(0.6 * W1[i] - 0.5 * A0[i]) else 0.0
          for i in range(N)]
    Y = [1.0 + 0.5 * W0[i] + 0.4 * W1[i]
         + A0[i] * (0.9 * W0[i] - 0.1)
         + A1[i] * (0.9 - 0.8 * W1[i])
         + 0.5 * rng.standard_normal() for i in range(N)]
    d0 = [1.0 if _v_dyn(1.0, w) - _v_dyn(0.0, w) > 0.0 else 0.0
          for w in W0]
    return {
        "W0": W0, "A0": A0, "W1": W1, "A1": A1, "Y": Y,
        "A": [[A0[i], A1[i]] for i in range(N)],
        "L": [[[W0[i]] for i in range(N)], [[W1[i]] for i in range(N)]],
        "d0_true": d0,
        "d1_true": [1.0 if W1[i] < CUT else 0.0 for i in range(N)],
        "psi": sum(_v_dyn(d0[i], W0[i]) for i in range(N)) / N,
        "static": {(a0, a1): sum(_v_static(a0, a1, w) for w in W0) / N
                   for a0 in (0.0, 1.0) for a1 in (0.0, 1.0)},
    }


def test_the_design_is_worth_estimating(two_stage):
    """If the optimal dynamic rule did not beat every static regime,
    every test below would pass for free."""
    d = two_stage
    assert d["psi"] > max(d["static"].values()) + 0.05
    assert 0.15 < sum(d["d0_true"]) / N < 0.85
    assert 0.15 < sum(d["d1_true"]) / N < 0.85


def test_backward_induction_recovers_both_rules(two_stage):
    """Theorem 22.1."""
    d = two_stage
    fit = sequential_blips(d["Y"], d["L"][0], d["A0"], d["L"][1],
                           d["A1"])
    a1 = sum(1 for i in range(N)
             if fit["d1"][int(d["A0"][i])][i] == d["d1_true"][i]) / N
    a0 = sum(1 for i in range(N)
             if fit["d0"][i] == d["d0_true"][i]) / N
    assert a1 > 0.97
    assert a0 > 0.93


def test_the_stage_one_contrast_carries_the_stage_two_rule(two_stage):
    """Contrasting A(0) at a FIXED A(1) -- the shortcut -- gives a
    materially different first-stage rule, so the test above is really
    testing backward induction."""
    d = two_stage
    for a1 in (0.0, 1.0):
        shortcut = [1.0 if _v_static(1.0, a1, w) - _v_static(0.0, a1, w)
                    > 0.0 else 0.0 for w in d["W0"]]
        agree = sum(1 for i in range(N)
                    if shortcut[i] == d["d0_true"][i]) / N
        assert agree < 0.93


def test_cv_tmle_recovers_the_mean_under_the_optimal_rule(two_stage):
    d = two_stage
    r = tmle_dynamic_regime(d["Y"], d["A"], d["L"], n_folds=5)
    assert r["estimate"] == pytest.approx(d["psi"], abs=0.07)
    assert r["ci"][0] <= d["psi"] <= r["ci"][1]
    assert r["estimate"] > r["best_static"]


def test_it_solves_the_cross_validated_eic_equation(two_stage):
    """Sec. 22.6 says exactly zero, not o_P(n^-1/2)."""
    d = two_stage
    r = tmle_dynamic_regime(d["Y"], d["A"], d["L"], n_folds=5)
    assert abs(r["eic_mean"]) < 1e-8


def test_the_static_comparators_match_their_closed_forms(two_stage):
    d = two_stage
    r = tmle_dynamic_regime(d["Y"], d["A"], d["L"], n_folds=5)
    for a0 in (0.0, 1.0):
        for a1 in (0.0, 1.0):
            key = "static_%d%d" % (int(a0), int(a1))
            assert r[key] == pytest.approx(d["static"][(a0, a1)],
                                           abs=0.07)


@pytest.mark.parametrize("method", ["cv-tmle", "tmle", "ipw", "gcomp"])
def test_every_route_lands_on_the_truth(two_stage, method):
    d = two_stage
    r = tmle_dynamic_regime(d["Y"], d["A"], d["L"], method=method,
                            n_folds=5)
    assert r["estimate"] == pytest.approx(d["psi"], abs=0.16)


def test_known_g_is_the_smart_case(two_stage):
    d = two_stage
    p0 = [expit(0.8 * d["W0"][i]) for i in range(N)]
    p1 = [expit(0.6 * d["W1"][i] - 0.5 * d["A0"][i]) for i in range(N)]
    r = tmle_dynamic_regime(d["Y"], d["A"], d["L"], known_g=(p0, p1),
                            n_folds=5)
    assert r["known_g"]
    assert r["estimate"] == pytest.approx(d["psi"], abs=0.07)


def test_a_supplied_regime_is_scored_not_optimised(two_stage):
    d = two_stage
    r_true = tmle_dynamic_regime(
        d["Y"], d["A"], d["L"],
        regime=(d["d0_true"], [d["d1_true"], d["d1_true"]]))
    assert r_true["rule_source"] == "supplied"
    assert r_true["estimate"] == pytest.approx(d["psi"], abs=0.07)
    # a bad rule must score badly, or the regime argument is ignored
    r_all = tmle_dynamic_regime(d["Y"], d["A"], d["L"],
                                regime=([1.0] * N, [1.0] * N))
    assert r_all["estimate"] == pytest.approx(d["static"][(1.0, 1.0)],
                                              abs=0.09)
    assert r_all["estimate"] < r_true["estimate"]


def test_an_exceptional_law_is_flagged():
    """Eq. (22.5): a blip that is flat at zero breaks the argmax, and
    that has to be reported rather than discovered later."""
    rng = np.random.default_rng(77)
    M = 1500
    W0 = [rng.standard_normal() for _ in range(M)]
    A0 = [1.0 if rng.uniform() < 0.5 else 0.0 for _ in range(M)]
    W1 = [0.5 * W0[i] + S_W1 * rng.standard_normal() for i in range(M)]
    A1 = [1.0 if rng.uniform() < 0.5 else 0.0 for _ in range(M)]
    Y = [1.0 + 0.5 * W0[i] + 0.4 * W1[i]
         + A1[i] * (0.9 - 0.8 * W1[i])
         + 0.5 * rng.standard_normal() for i in range(M)]
    r = tmle_dynamic_regime(Y, [[A0[i], A1[i]] for i in range(M)],
                            [[[W0[i]] for i in range(M)],
                             [[W1[i]] for i in range(M)]], n_folds=5)
    assert r["exceptional_share_1"] > 0.5
    assert r["exceptional_share_2"] < 0.2


def test_exceptional_law_share_counts_what_it_says():
    assert exceptional_law_share([0.0, 0.005, 0.5, -0.9],
                                 tol=0.01) == 0.5
    assert exceptional_law_share([1.0, -1.0]) == 0.0


def test_positivity_is_reported(two_stage):
    d = two_stage
    r = tmle_dynamic_regime(d["Y"], d["A"], d["L"], n_folds=5)
    assert r["max_weight"] > 1.0
    assert 0.0 < r["min_g0"] < 1.0
    assert 0.0 < r["min_g1"] < 1.0


def test_argument_checks(two_stage):
    d = two_stage
    with pytest.raises(ValueError):
        tmle_dynamic_regime(d["Y"], d["A"], d["L"], method="nope")
    with pytest.raises(ValueError):
        tmle_dynamic_regime(d["Y"], [[2.0, 0.0]] * N, d["L"])
    with pytest.raises(ValueError):
        tmle_dynamic_regime([1.0] * N, d["A"], d["L"])
    with pytest.raises(ValueError):
        tmle_dynamic_regime(d["Y"], d["A"], [d["L"][0]])
    with pytest.raises(ValueError):
        tmle_dynamic_regime(d["Y"], [[1.0]] * N, d["L"])
    with pytest.raises(ValueError):
        tmle_dynamic_regime(d["Y"], d["A"], d["L"], regime=[1.0, 0.0])
    with pytest.raises(ValueError):
        intervention_mechanism(d["L"][0], d["A0"], d["L"][1], d["A1"],
                               trim=0.9)
