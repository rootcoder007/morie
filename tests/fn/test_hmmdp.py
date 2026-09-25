"""Tests for hmmdp.geron_mdp (value iteration on a finite MDP)."""

import itertools

import pytest

from morie.fn.hmmdp import geron_mdp

# two states, two actions; P[s][a] is the next-state distribution
P = [[[0.7, 0.3], [0.1, 0.9]], [[0.4, 0.6], [0.95, 0.05]]]
R = [[1.0, 0.0], [-0.5, 2.0]]
G = 0.9


def _policy_value(pol, r):
    # V = (I - G P_pi)^-1 r_pi for a 2-state chain, by Cramer's rule
    a = [[(1.0 if i == j else 0.0) - G * P[i][pol[i]][j] for j in range(2)]
         for i in range(2)]
    b = [r[i][pol[i]] for i in range(2)]
    det = a[0][0] * a[1][1] - a[0][1] * a[1][0]
    return [(b[0] * a[1][1] - a[0][1] * b[1]) / det,
            (a[0][0] * b[1] - b[0] * a[1][0]) / det]


def _optimum(r):
    vals = {pol: _policy_value(pol, r) for pol in itertools.product(range(2), repeat=2)}
    best = max(vals, key=lambda p: sum(vals[p]))
    for p in vals:                       # the optimum dominates every policy
        assert all(vals[best][i] >= vals[p][i] - 1e-12 for i in range(2))
    return best, vals[best]


def test_hmmdp_basic():
    """V equals the exact value of the best of the four deterministic
    policies. Value iteration stops when a sweep moves V by < tol, so
    |V - V*| <= tol * G / (1 - G) = 9e-13; 1e-11 leaves room for the
    rounding of the ~300 sweeps."""
    best, v = _optimum(R)
    result = geron_mdp(["s0", "s1"], ["a0", "a1"], P, R, gamma=G, tol=1e-13)
    assert isinstance(result, dict)
    assert [float(x) for x in result["V"]] == pytest.approx(v, rel=0, abs=1e-11)
    assert [int(a) for a in result["policy"]] == list(best)


def test_hmmdp_edge():
    """(S, A, S') rewards reduce to r(s, a) = sum_s' P(s'|s,a) R(s,a,s');
    a leaking transition row is refused."""
    R3 = [[[2.0, -1.0], [0.0, 0.5]], [[1.0, -3.0], [4.0, 0.0]]]
    r = [[sum(P[s][a][t] * R3[s][a][t] for t in range(2)) for a in range(2)]
         for s in range(2)]
    best, v = _optimum(r)
    res = geron_mdp(["s0", "s1"], ["a0", "a1"], P, R3, gamma=G, tol=1e-13)
    assert [float(x) for x in res["V"]] == pytest.approx(v, rel=0, abs=1e-11)
    with pytest.raises(ValueError):
        geron_mdp(["s0", "s1"], ["a0", "a1"], [[[0.5, 0.4], [0.1, 0.9]], P[1]], R, gamma=G)


# --- appended: the module's own worked example as a gate -----------

import doctest as _doctest

import morie.fn.hmmdp as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
