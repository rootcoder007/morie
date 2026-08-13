"""Tests for itrgrf. Full anchor: ledger/wave3/anchor_grf_family.py."""
import pytest
from morie.fn import _s03core as k
from morie.fn.itrgrf import itr_forest, policy_from_tau
from ._grf_fixture import confounded


def test_the_rule_is_the_threshold_rule():
    assert policy_from_tau([-1.0, 0.5, 2.0], cost=1.0) == [0.0, 0.0, 1.0]
    assert policy_from_tau([-1.0, 0.5, 2.0]) == [0.0, 1.0, 1.0]


def test_in_sample_evaluation_claims_a_gain_that_is_not_there():
    """The rule is an argmax, so scoring it on the data that produced it
    inherits the winner's curse. One draw cannot separate the two -- both
    are noise around zero -- so the claim is about the AVERAGE with no
    effect present anywhere."""
    gi, gs = [], []
    for rep in range(5):
        d = confounded(400, 700 + rep, tau_scale=0.0)
        gi.append(itr_forest(d["y"], d["W"], d["X"], n_trees=60,
                             min_leaf=5, seed=rep,
                             evaluate="in-sample",
                             propensity=d["e"])["gain_over_treat_none"])
        gs.append(itr_forest(d["y"], d["W"], d["X"], n_trees=60,
                             min_leaf=5, seed=rep, evaluate="split",
                             propensity=d["e"])["gain_over_treat_none"])
    assert k.mean(gi) > k.mean(gs)


def test_with_a_real_effect_the_rule_pays():
    d = confounded(600, 34, tau_scale=1.0)
    r = itr_forest(d["y"], d["W"], d["X"], n_trees=80, min_leaf=5,
                   seed=5, propensity=d["e"])
    assert r["gain_over_treat_none"] > 0.0
    # tau = 0.5 + X2 is positive for about 69 per cent of the
    # population, so the rule should treat most but not all
    assert 0.4 < r["treated_fraction"] < 0.99
    costly = itr_forest(d["y"], d["W"], d["X"], cost=2.0, n_trees=80,
                        min_leaf=5, seed=5, propensity=d["e"])
    assert costly["treated_fraction"] < r["treated_fraction"]


def test_argument_checks():
    d = confounded(200, 35)
    with pytest.raises(ValueError):
        itr_forest(d["y"], [2.0] * d["n"], d["X"])
    with pytest.raises(ValueError):
        itr_forest(d["y"], d["W"], d["X"], evaluate="nope")
    with pytest.raises(ValueError):
        itr_forest(d["y"][:30], d["W"][:30], d["X"][:30])
