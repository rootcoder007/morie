"""Tests for mdppol.mdp_policy_iteration.

Anchors: exact rational Bellman solve by policy enumeration (borrowed
from test_mdpval, an independent route from both iterative algorithms)
and agreement with value iteration.
"""

from morie.fn.mdppol import mdppol
from morie.fn.mdpval import mdpval

from .test_mdpval import P, R, GAMMA, _brute_optimal


def test_mdppol_closed_form_two_state():
    P2 = [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]]
    R2 = [[1.0, 0.0], [0.0, 0.0]]
    r = mdppol(P2, R2, 0.5)
    assert abs(r["estimate"][0] - 2.0) < 1e-10
    assert r["policy"][0] == 0.0
    assert r["policy_stable"]


def test_mdppol_matches_exact_policy_enumeration():
    exact = _brute_optimal()
    r = mdppol(P, R, GAMMA, tol=1e-13)
    for s in range(3):
        assert abs(r["estimate"][s] - float(exact[s])) < 1e-9


def test_mdppol_agrees_with_value_iteration():
    a = mdppol(P, R, GAMMA, tol=1e-13)
    b = mdpval(P, R, GAMMA, tol=1e-13)
    for s in range(3):
        assert abs(a["estimate"][s] - b["estimate"][s]) < 1e-9
        assert a["policy"][s] == b["policy"][s]


def test_mdppol_terminates_despite_ties():
    # two identical actions everywhere: literal box can flip forever
    P2 = [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]
    R2 = [[1.0, 1.0], [0.0, 0.0]]
    r = mdppol(P2, R2, 0.5)
    assert r["policy_stable"]
    assert r["n_improve"] <= 3
