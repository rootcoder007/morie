"""Tests for thrtmt -- optimal ITR under a resource constraint.

Replaces a generated test that called a stub returning mean(y). Full
anchor: ledger/wave3/anchor_thrtmt.py.
"""

import pytest

from morie.fn import _array_core as np
from morie.fn.thrtmt import (blip_function, rc_rule, rc_threshold,
                             rule_value, threshold_treatment_msm)

N = 1500


@pytest.fixture(scope="module")
def harmful_for_some():
    """The blip is 0.9 * W1, so treatment helps when W1 > 0 and harms
    when W1 < 0 -- about half the population should never be treated
    however much capacity there is."""
    rng = np.random.default_rng(9)
    W1 = [rng.standard_normal() for _ in range(N)]
    W2 = [rng.standard_normal() for _ in range(N)]
    A = [1.0 if rng.uniform() < 0.5 else 0.0 for _ in range(N)]
    Y = [0.5 + 0.9 * W1[i] * A[i] + 0.4 * W2[i]
         + 0.3 * rng.standard_normal() for i in range(N)]
    W = [[W1[i], W2[i]] for i in range(N)]
    blip, info = blip_function(Y, A, W)
    return {"Y": Y, "A": A, "W": W, "blip": blip,
            "q1": info["q1"], "q0": info["q0"]}


def test_the_budget_is_respected(harmful_for_some):
    d = harmful_for_some
    for kap in (0.05, 0.1, 0.25, 0.4):
        _, ri = rc_rule(d["blip"], kap)
        assert ri["treated_fraction"] <= kap + 1e-9


def test_slack_budget_is_clipped_at_zero(harmful_for_some):
    """tau = max{eta, 0}: spare capacity must not go to people the
    treatment harms."""
    d = harmful_for_some
    r = threshold_treatment_msm(d["Y"], d["A"], d["W"], kappa=0.8)
    assert r["eta"] < 0.0
    assert r["tau"] == 0.0
    assert r["treated_fraction"] < 0.6      # not the permitted 0.8
    assert not any(d["blip"][i] < 0.0 and r["rule"][i] > 0.0
                   for i in range(N))
    # with the budget slack it IS the unconstrained optimum
    assert r["treated_fraction"] == pytest.approx(
        r["unconstrained_fraction"], abs=1e-12)


def test_no_budget_respecting_rule_beats_it(harmful_for_some):
    """Theorem 1's optimality claim, against random competitors and
    against the same budget spent on the worst responders."""
    d = harmful_for_some
    kap = 0.25
    d_opt, _ = rc_rule(d["blip"], kap)
    v_opt = rule_value(d["q1"], d["q0"], d_opt)
    for trial in range(100):
        rr = np.random.default_rng(1000 + trial)
        cand = [1.0 if float(rr.uniform()) < kap else 0.0
                for _ in range(N)]
        if sum(cand) / N > kap + 1e-9:
            continue
        assert rule_value(d["q1"], d["q0"], cand) <= v_opt + 1e-12
    worst = set(sorted(range(N), key=lambda i: d["blip"][i])[:int(kap * N)])
    d_bad = [1.0 if i in worst else 0.0 for i in range(N)]
    assert rule_value(d["q1"], d["q0"], d_bad) < v_opt


def test_value_is_monotone_in_the_budget(harmful_for_some):
    d = harmful_for_some
    vals = [threshold_treatment_msm(d["Y"], d["A"], d["W"],
                                    kappa=kp)["value"]
            for kp in (0.05, 0.1, 0.25, 0.4, 0.5)]
    assert all(vals[i + 1] >= vals[i] - 1e-9
               for i in range(len(vals) - 1))
    r = threshold_treatment_msm(d["Y"], d["A"], d["W"], kappa=0.9)
    assert all(v <= r["value_unconstrained"] + 1e-9 for v in vals)
    # treating everyone is worse than treating the right people
    assert r["value_treat_all"] < r["value_unconstrained"]


def test_argument_checks(harmful_for_some):
    d = harmful_for_some
    with pytest.raises(ValueError):
        rc_threshold(d["blip"], 0.0)
    with pytest.raises(ValueError):
        rc_threshold(d["blip"], 1.0)
    with pytest.raises(ValueError):
        rc_rule(d["blip"], 0.2, rule="nope")
    with pytest.raises(ValueError):
        blip_function(d["Y"], [2.0] * N, d["W"])
