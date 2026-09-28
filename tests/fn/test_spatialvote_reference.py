"""spatialvote: majority rule, pivots, probabilistic voting, competition and bargaining against closed forms."""

import math

import pytest

from morie.fn.spatialvote import (
    baron_ferejohn,
    candidate_equilibrium,
    hotelling_price_location,
    kalai_smorodinsky,
    majority_compare,
    pareto_set,
    pca_ideal_points,
    pivot_points,
    probabilistic_vote,
    salop_circle,
    vote_trading_riker_brams,
    win_set,
)


def test_majority_and_pivots():
    ideals = [(0.0, 0.0), (4.0, 0.0), (2.0, 3.0)]
    r = majority_compare(ideals, (2.0, 1.0), (3.5, 3.0), weights=[1, 1, 1])
    assert (r.votes_x, r.votes_y, r.winner) == (2.0, 1.0, "x")
    s = majority_compare([(0.0, 0.0), (1.0, 1.0)], (0.0, 1.0), (1.0, 0.0), A=[1.0, 9.0])
    # weighted distance: voter 0 prefers (1, 0) (cost 1 vs 9), voter 1 prefers (0, 1)
    assert (s.votes_x, s.votes_y) == (1.0, 1.0)
    p = pivot_points(
        [1, 2, 3, 4, 5, 6, 7],
        weights=[1, 1, 1, 1, 1, 1, 4],
        supermajority=0.6,
        veto=2 / 3,
        chambers=[[0, 1, 2], [3, 4, 5, 6]],
    )
    # cumulative weights 1, 2, 3, 4, 5, 6, 10 of 10
    assert p.median == 5.0 and p.supermajority_pivots == [4.0, 6.0] and p.veto_pivot == 4.0
    assert p.chamber_medians == [2.0, 7.0] and p.bicameral_core == [2.0, 7.0] and p.gridlock == [2.0, 7.0]
    ps = pareto_set([(0, 0), (4, 0), (0, 4), (1, 1)], [(1, 1), (3, 3), (2, 2)])
    assert ps.inside == [True, False, True]
    ws = win_set(ideals, (2.0, 1.0), [(2.0, 0.5), (2.0, 2.0), (5.0, 5.0)])
    assert ws.in_win_set == [
        majority_compare(ideals, g, (2.0, 1.0)).winner == "x" for g in [(2.0, 0.5), (2.0, 2.0), (5.0, 5.0)]
    ]


def test_probabilistic_vote():
    ideals = [(0.0,), (1.0,), (3.0,)]
    pos = [(0.5,), (2.0,)]
    for err, F in (
        ("logit", lambda z: 1 / (1 + math.exp(-z))),
        ("cauchy", lambda z: 0.5 + math.atan(z) / math.pi),
        ("laplace", lambda z: 0.5 * math.exp(z) if z < 0 else 1 - 0.5 * math.exp(-z)),
    ):
        r = probabilistic_vote(ideals, pos, valence=[0.3, 0.0], beta=0.7, error=err)
        want = [F(0.3 - 0.7 * ((x - 0.5) ** 2 - (x - 2.0) ** 2)) for (x,) in ideals]
        assert [p[0] for p in r.probabilities] == pytest.approx(want)
        assert r.shares[0] == pytest.approx(sum(want) / 3)
    m = probabilistic_vote(ideals, [(0.0,), (1.0,), (2.0,)])
    for p, (x,) in zip(m.probabilities, ideals):
        u = [-((x - c) ** 2) for c in (0.0, 1.0, 2.0)]
        e = [math.exp(v) for v in u]
        assert p == pytest.approx([v / sum(e) for v in e])
    with pytest.raises(ValueError):
        probabilistic_vote(ideals, [(0.0,), (1.0,), (2.0,)], error="probit")


def test_competition_and_bargaining():
    g = [(k / 20,) for k in range(21)]
    eq = candidate_equilibrium([(0.1,), (0.3,), (0.8,), (0.6,)], g, beta=0.3)
    assert eq.converged and eq.positions[0] == eq.positions[1] == (0.45,)  # mean voter 0.45
    val = candidate_equilibrium([(0.1,), (0.3,), (0.8,), (0.6,)], g, beta=4.0, valence=(0.5, 0.0))
    assert val.distance >= 0
    h = hotelling_price_location(0.1, 0.2, t=2.0, c=1.0)
    L = 0.7
    assert h.prices == pytest.approx([1 + 2 * L * (1 - 0.1 / 3), 1 + 2 * L * (1 + 0.1 / 3)])
    assert h.demand[0] + h.demand[1] == pytest.approx(1.0)
    s = salop_circle(3, t=1.5, c=0.2, fixed_cost=0.1, length=2.0)
    assert (s.price, s.profit) == pytest.approx((0.2 + 1.0, 1.5 * 4 / 9 - 0.1))
    ks = kalai_smorodinsky([(0.0, 2.0), (1.0, 1.5), (2.0, 0.0)])
    assert ks.ideal == [2.0, 2.0] and ks.kalai_smorodinsky == pytest.approx([1.2, 1.2])
    # Nash on the first segment u2 = 2 - u1/2: max u1 (2 - u1/2) at u1 = 2 lies outside; second segment 1.5 - 1.5(u1 - 1)
    assert ks.nash == pytest.approx([1.0, 1.5])
    bf = baron_ferejohn(7, 0.9)
    assert bf.proposer_share == pytest.approx(1 - 3 * 0.9 / 7) and bf.coalition_size == 4


def test_vote_trading_and_pca():
    r = vote_trading_riker_brams([[3, -1], [-1, 3], [-3, -3]])
    assert (r.sincere, r.after_trade, r.welfare_before, r.welfare_after) == ([False, False], [True, True], 0.0, -2.0)
    none = vote_trading_riker_brams([[1, 1], [1, 1], [-1, -1]])
    assert none.trades == [] and none.sincere == none.after_trade
    p = pca_ideal_points([[1, 1, 0, 1], [1, 1, 0, None], [0, 0, 1, 0], [0, 1, 1, 0]])
    s = [v[0] for v in p.scores]
    assert s[0] <= 0 and s[0] * s[2] < 0 and 0 < p.variance_share[0] <= 1
