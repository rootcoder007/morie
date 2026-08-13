"""Tests for ipwgrf. Full anchor: ledger/wave3/anchor_grf_family.py."""
import pytest
from morie.fn import _s03core as k
from morie.fn.ipwgrf import aipw_scores, ipw_forest
from ._grf_fixture import confounded


@pytest.fixture(scope="module")
def d():
    # randomized: the score is identified without asking the
    # forest to recover a propensity it cannot at this n
    return confounded(700, 41, randomized=True)


def test_the_score_is_the_textbook_expression():
    g, _ = aipw_scores([1.0], [1.0], [0.5], [0.2], [0.5])
    assert g[0] == pytest.approx(0.5 - 0.2 + (1.0 - 0.5) / 0.5,
                                 abs=1e-12)


def test_double_robustness_survives_one_broken_nuisance(d):
    truth = k.mean(d["tau"])
    both = ipw_forest(d["y"], d["W"], d["X"], n_trees=80, min_leaf=5,
                      seed=6)
    bo = ipw_forest(d["y"], d["W"], d["X"], n_trees=80, min_leaf=5,
                    seed=6, break_outcome=True)
    bp = ipw_forest(d["y"], d["W"], d["X"], n_trees=80, min_leaf=5,
                    seed=6, break_propensity=True)
    assert abs(both["estimate"] - truth) < 3.0 * both["se"]
    assert abs(bo["estimate"] - truth) < 4.0 * bo["se"]
    assert abs(bp["estimate"] - truth) < 4.0 * bp["se"]
    # with the outcome model flattened there is nothing left to plug in
    assert abs(bo["plug_in"]) < 1e-12


def test_overlap_is_reported(d):
    r = ipw_forest(d["y"], d["W"], d["X"], n_trees=80, min_leaf=5,
                   seed=6)
    assert r["max_weight"] > 1.0
    assert 0.0 < r["min_propensity"] <= r["max_propensity"] < 1.0


def test_argument_checks(d):
    with pytest.raises(ValueError):
        ipw_forest(d["y"], [1.0] * d["n"], d["X"])
    with pytest.raises(ValueError):
        ipw_forest(d["y"], [2.0] * d["n"], d["X"])
    with pytest.raises(ValueError):
        ipw_forest(d["y"], d["W"], d["X"], trim=0.8)
    with pytest.raises(ValueError):
        ipw_forest(d["y"][:30], d["W"][:30], d["X"][:30])
