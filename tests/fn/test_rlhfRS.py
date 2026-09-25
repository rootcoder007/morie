"""Tests for rlhfRS.rlhf_recommendation (session MDP + off-policy value)."""

import math

import pytest

from morie.fn.rlhfRS import rlhf_recommendation


ENV = {"transition": [[[0.0, 1.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]],
       "reward": [[1.0, 0.0], [0.5, 2.0]], "start": 0,
       "log": [(0, 0, 1.0), (0, 1, 0.0), (1, 1, 2.0), (1, 0, 0.5)],
       "behaviour": [0.5, 0.5, 0.25, 0.75]}
PI = [[1.0, 0.0], [0.0, 1.0]]


def test_rlhfRS_basic():
    """Deterministic MDP: state 0 -a0-> state 1 -a1-> state 1, so the
    return is 1 + sum_{t>=1} 2 gamma^t with zero spread.  IPS is
    mean(w r), SNIPS sum(w r)/sum(w), DR adds the model baseline, with
    w = pi(a|s)/mu; ESS = (sum w)^2 / sum w^2; NDCG/P@k/MRR by definition."""
    g, H = 0.9, 5
    r = rlhf_recommendation(ENV, PI, n_episodes=4, horizon=H, gamma=g,
                            relevance=[0, 3, 0, 1], k=3)
    ret = 1.0 + sum(2.0 * g ** t for t in range(1, H))
    assert r["returns"] == [pytest.approx(ret, rel=1e-15)] * 4
    assert r["se"] == pytest.approx(0.0, abs=1e-15)
    w = [2.0, 0.0, 4.0, 0.0]
    assert r["weights"] == w
    assert r["off_policy"] == pytest.approx((2.0 * 1.0 + 4.0 * 2.0) / 4, rel=1e-15)
    assert r["ess"] == pytest.approx(36.0 / 20.0, rel=1e-15)
    sn = rlhf_recommendation({"log": ENV["log"], "behaviour": ENV["behaviour"]}, PI, estimator="snips")
    assert sn["off_policy"] == pytest.approx(10.0 / 6.0, rel=1e-15)
    q = [[0.8, 0.1], [0.4, 1.5]]
    dr = rlhf_recommendation({"log": ENV["log"], "behaviour": ENV["behaviour"]}, PI, estimator="dr", reward_model=q)
    want = sum(q[s][PI[s].index(1.0)] + wi * (rr - q[s][a]) for (s, a, rr), wi in zip(ENV["log"], w)) / 4
    assert dr["off_policy"] == pytest.approx(want, rel=1e-15)
    dcg = 3 / math.log2(3)
    idcg = 3 / math.log2(2) + 1 / math.log2(3)
    assert r["ndcg_at_k"] == pytest.approx(dcg / idcg, rel=1e-15)
    assert r["precision_at_k"] == pytest.approx(1 / 3, rel=1e-15)
    assert r["mrr"] == 0.5


def test_rlhfRS_edge():
    """No model and no log raises; so does a target action the logger
    could never take (the support condition)."""
    with pytest.raises(ValueError):
        rlhf_recommendation({}, PI)
    with pytest.raises(ValueError):
        rlhf_recommendation({"log": [(0, 0, 1.0)], "behaviour": [0.0]}, PI)
