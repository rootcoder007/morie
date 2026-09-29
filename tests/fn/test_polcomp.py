"""Tests for polcomp: spatial party competition and legislative bargaining."""

import math

from morie.fn.polcomp import (
    best_response_equilibrium,
    committee_outcome,
    effective_number_parties,
    entry_game,
    laver_dynamics,
    median_party,
    merge_split_parties,
    rubinstein_bargaining,
    setter_outcome,
    strategic_vote_shares,
    vote_shares,
)

V = [math.sin(i * 1.7) * 0.5 + (0.6 if i % 3 == 0 else -0.2) for i in range(31)]


def test_proximity_and_logit_shares():
    s = vote_shares([-0.3, 0.1, 0.6], V)
    want = [0, 0, 0]
    for v in V:
        d = [abs(v - p) for p in (-0.3, 0.1, 0.6)]
        want[d.index(min(d))] += 1
    assert s == [w / 31 for w in want]
    lg = vote_shares([0.0, 1.0], [0.25], rule="logit", beta=2.0)
    assert abs(lg[0] - 1 / (1 + math.exp(-2 * (0.75**2 - 0.25**2)))) <= 1e-15


def test_two_party_best_response_is_the_median():
    grid = [i / 20 - 1 for i in range(41)]
    r = best_response_equilibrium(V, 2, grid)
    med = sorted(V)[15]
    assert r.converged and all(abs(p[0] - med) <= 0.05 for p in r.positions)


def test_strategic_voting_concentrates_on_viable_parties():
    r = strategic_vote_shares([-0.5, 0.0, 0.4, 0.8], V)
    assert len(r.viable) == 2 and abs(sum(r.strategic) - 1) <= 1e-12
    assert all(r.strategic[j] == 0.0 for j in range(4) if j not in r.viable)


def test_entry_game_entrant_best_responds():
    grid = [i / 5 - 1 for i in range(11)]
    r = entry_game(V, grid)
    best = max(vote_shares(r.incumbents + [g], V)[-1] for g in [[x] for x in grid])
    assert abs(r.shares[-1] - best) <= 1e-12


def test_indices_and_bargaining():
    e = effective_number_parties([10, 25, 7, 3])
    p = [10 / 45, 25 / 45, 7 / 45, 3 / 45]
    assert abs(e.laakso_taagepera - 1 / sum(v * v for v in p)) <= 1e-12
    b = rubinstein_bargaining(0.8, 0.95)
    assert abs(b.proposer - 0.05 / (1 - 0.76)) <= 1e-12
    assert median_party([0.3, -1.0, 0.8, 0.1], [20, 30, 25, 25]).index == 3


def test_laver_aggregator_moves_to_centroid_and_merge_split():
    v = [(-1.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.2, 0.9)]
    r = laver_dynamics(v, [(0.1, 0.8), (-2.0, -2.0)], ["aggregator", "sticker"], 1)
    sup = [p for p in v if math.dist(p, (0.1, 0.8)) < math.dist(p, (-2.0, -2.0))]
    assert r.path[1][0] == [sum(p[0] for p in sup) / len(sup), sum(p[1] for p in sup) / len(sup)]
    m = merge_split_parties([-0.5, 0.0, 0.5], V, merge=(0, 1))
    assert abs(sum(m.new) - 1) <= 1e-12 and len(m.new) == 2


def test_setter_and_committee():
    assert setter_outcome(0.9, 0.5, 0.3).outcome == 0.7
    assert setter_outcome(0.6, 0.5, 0.3).outcome == 0.6
    c = committee_outcome([0.1, 0.2, 0.3], [0.1, 0.3, 0.5, 0.6, 0.9], 0.2)
    assert c.outcome == 0.2 and not c.gate_opened
